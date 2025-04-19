import argparse
import os
import math
import sys

sys.path.append('./taming-transformers')
from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from CLIP import clip
import kornia.augmentation as K
import numpy as np
import imageio
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cur_iter = 0


def noise_gen(shape):
    n, c, h, w = shape
    noise = torch.zeros([n, c, 1, 1])
    for i in reversed(range(5)):
        h_cur, w_cur = h // 2 ** i, w // 2 ** i
        noise = F.interpolate(noise, (h_cur, w_cur), mode='bicubic', align_corners=False)
        noise += torch.randn([n, c, h_cur, w_cur]) / 5
    return noise


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


def vector_quantize(z, codebook):
    d = z.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * z @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    replace_grad = ReplaceGrad.apply
    return replace_grad(x_q, z)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        replace_grad = ReplaceGrad.apply
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2, p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7))
        self.noise_fac = 0.1

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


def load_vqgan_model(config_path, checkpoint_path):
    config_path = os.path.join('.', 'checkpoints', config_path)
    checkpoint_path = os.path.join('.', 'checkpoints', checkpoint_path)
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
        gumbel = True
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)
    return image.resize(size, Image.LANCZOS)


class VQGANPainterWithMSE:
    def __init__(self, args):
        # mse_weight为当前正则化损失的权重
        self.mse_weight = args.init_weight
        self.mse_decay = 0
        if args.init_weight:
            # 设置正则化损失的衰减速率，每次衰减mse_decay
            self.mse_decay = args.init_weight / args.mse_epoches

        self.model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
        self.perceptor = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
        self.args = args

        cut_size = self.perceptor.visual.input_resolution
        e_dim = self.model.quantize.e_dim
        # 共有num_resolution-1个上采样层
        # z中1个token对应最终图像中1块 f*f (在当前网络中，f=16)像素大小的区域
        f = 2 ** (self.model.decoder.num_resolutions - 1)
        self.make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
        n_toks = self.model.quantize.n_e
        toksX, toksY = args.size[0] // f, args.size[1] // f
        # 生成图像的大小会向下取整为f的倍数
        sideX, sideY = toksX * f, toksY * f
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        if args.init_image:
            pil_image = Image.open(args.init_image).convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            self.z, *_ = self.model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
        else:
            one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
            self.z = one_hot @ self.model.quantize.embedding.weight
            self.z = self.z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        # self.z_orig = self.z.clone()

        if args.mse_withzeros and not args.init_image:
            # 将 self.z_orig 初始化为与 self.z 相同形状的全零张量
            # 计算正则项时，将z与全零向量比较，计算对应损失
            self.z_orig = torch.zeros_like(self.z)
        else:
            # 若mse_withzeros为假，或者提供了初始图像，则z_orig为初始图像对应的隐变量
            self.z_orig = self.z.clone()

        self.z.requires_grad_(True)
        self.opt = optim.Adam([self.z], lr=args.step_size, weight_decay=0.00000000)

        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

        self.pMs = []

        for prompt in args.prompts:
            txt, weight, stop = parse_prompt(prompt)
            embed = self.perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(device))

        for prompt in args.image_prompts:
            path, weight, stop = parse_prompt(prompt)
            img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
            batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
            embed = self.perceptor.encode_image(self.normalize(batch)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(device))

        for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(generator=gen)
            self.pMs.append(Prompt(embed, weight).to(device))

    def synth(self, z):
        """
        :param z:
        :return: 0-1 之间的图像
        """
        clamp_with_grad = ClampWithGrad.apply

        z_q = vector_quantize(z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    def checkin(self, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        tqdm.write(f'i: {cur_iter}, loss: {sum(losses).item():g}, losses: {losses_str}')
        out = self.synth(self.z)
        TF.to_pil_image(out[0].cpu()).save('progress.png')
        # todo 显示图片

    def ascend_txt(self):
        out = self.synth(self.z)
        image_embed = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()

        result = []

        if self.args.init_weight:
            # 向损失中加入L2正则化，使得生成的图像更加稳定和自然
            result.append(F.mse_loss(self.z, self.z_orig) * self.mse_weight / 2)

            with torch.no_grad():
                if cur_iter > 0 and cur_iter % self.args.mse_decay_rate == 0 and cur_iter <= self.args.mse_decay_rate * self.args.mse_epoches:
                    # 在权重衰减区间内，模型每迭代mse_decay_rate，便衰减一次权重
                    if self.mse_weight - self.mse_decay > 0:
                        # 若权重仍可衰减
                        self.mse_weight = self.mse_weight - self.mse_decay
                        print(f"updated mse weight: {self.mse_weight}")
                    else:
                        self.mse_weight = 0
                        print(f"updated mse weight: {self.mse_weight}")

        for prompt in self.pMs:
            result.append(prompt(image_embed))
        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:, :, :]
        img = np.transpose(img, (1, 2, 0))
        filename = f"static/steps/{cur_iter + 1:04}.png"
        imageio.imwrite(filename, np.array(img))

        return result

    def train_once(self):
        self.opt.zero_grad()
        lossAll = self.ascend_txt()
        # if cur_iter % self.args.display_freq == 0:
        #     self.checkin(lossAll)
        loss = sum(lossAll)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

    def train(self, threadCTL):
        global cur_iter
        # cur_iter表示当前迭代的次数，也可以表示当前已生成的图片
        cur_iter = 0
        try:
            with tqdm(self.args.max_iterations) as pbar:
                isInterrupt = threadCTL['interrupt']
                while not isInterrupt:
                    self.train_once()
                    cur_iter += 1
                    if cur_iter == self.args.max_iterations:
                        break
                    pbar.update()
                    isInterrupt = threadCTL['interrupt']
                threadCTL['lastIter'] = cur_iter

        except KeyboardInterrupt:
            pass

    @staticmethod
    def getCurIter():
        return cur_iter


if __name__ == '__main__':
    # prompts = "a boy with a dog in the sunshine"
    # prompts = "A boy is running in the rain"
    prompts = "evening"
    width = 512
    height = 512
    model_selected = "vqgan_imagenet_f16_16384"  # ["vqgan_imagenet_f16_16384", "vqgan_imagenet_f16_1024", "wikiart_16384", "coco", "faceshq", "sflckr"]
    display_frequency = 50
    initial_image = ""
    target_images = ""
    seed = -1
    max_iterations = 700
    input_images = ""

    model_names = {"vqgan_imagenet_f16_16384": 'ImageNet 16384', "vqgan_imagenet_f16_1024": "ImageNet 1024",
                   "wikiart_16384": "WikiArt 16384", "coco": "COCO-Stuff", "faceshq": "FacesHQ", "sflckr": "S-FLCKR"}
    model_name = model_names[model_selected]
    # model_selected = 'vqgan_gumbel_f8_8192'

    if seed == -1:
        seed = None
    if initial_image == "None":
        initial_image = None
    if target_images == "None" or not target_images:
        target_images = []
    else:
        target_images = target_images.split("|")
        target_images = [image.strip() for image in target_images]

    if initial_image or target_images != []:
        input_images = True

    prompts = [frase.strip() for frase in prompts.split("|")]
    if prompts == ['']:
        prompts = []

    args = argparse.Namespace(
        prompts=prompts,
        image_prompts=target_images,
        noise_prompt_seeds=[],
        noise_prompt_weights=[],
        size=[width, height],
        init_image=initial_image,
        init_weight=1.5,
        clip_model='ViT-B/32',
        vqgan_config=f'{model_selected}.yaml',
        vqgan_checkpoint=f'{model_selected}.ckpt',
        step_size=0.95,
        cutn=64,
        cut_pow=1.,
        display_freq=display_frequency,
        seed=seed,
        max_iterations=max_iterations,

        # mse settings
        mse_withzeros=True,
        mse_decay_rate=50,
        mse_epoches=5,
    )

    print('Using device:', device)
    if prompts:
        print('Using text prompt:', prompts)
    if target_images:
        print('Using image prompts:', target_images)
    if args.seed is None:
        seed = torch.seed()
    else:
        seed = args.seed
    torch.manual_seed(seed)
    print('Using seed:', seed)

    painter = VQGANPainterWithMSE(args)

    threadCTL = {
        'imageGenerated': False,  # 完成图像生成 或 图像生成终止
        'interrupt': False,  # 终止图像生成
        'videoGenerated': False,  # 完成视频生成
        'finish': False,  # 线程任务结束（ 生成终止 或 完成图像、视频的生成)
        'lastIter': 0,  # 图像的最后迭代次数
    }

    painter.train(threadCTL)
