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

import kornia.augmentation as K
import numpy as np
import imageio
from PIL import ImageFile, Image

from models import Discriminator

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cur_iter = 0


def sinc(x):
    """
    定义sinc函数
    sinc(x) = sin(π * x) / (π * x)   if x != 0
    sinc(x) = 1                      if x == 0
    """
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    """
    定义了 Lanczos 滤波器，用于信号处理中的采样插值, 并将计算得到的结果进行归一化
    lanczos(x, a) = sinc(x) * sinc(x / a)  if -a < x < a
    lanczos(x, a) = 0                      其他情况
    """
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    """
    ramp函数用于生成一个线性递增的序列,其中比例由ratio决定，宽度由width决定
    :param ratio: 序列中的元素每次递增ratio
    :param width: 序列的宽度为[-_width,_width]，其中_width为ratio的整数倍，通过对width向上取整得到
    """
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    """
    resample用于图像重采样，size为目标图像的大小
    先通过 Lanczos 插值核进行一次卷积操作，然后再使用双三次插值进行二次插值
    """
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
    """
    实现替换梯度替换
    用于在反向传播过程中将x_backward的梯度替换为x_forward的梯度
    通过继承 torch.autograd.Function 类，并实现其中的静态方法 forward 和 backward，可以定义一个具有自定义操作和梯度计算的函数。
    forward 方法定义了前向传播的逻辑，接收输入张量并计算输出张量。
    backward 方法定义了反向传播的逻辑，接收输入张量的梯度，计算相应的梯度随后反向传播。
    """

    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


class ClampWithGrad(torch.autograd.Function):
    """
    实现梯度裁剪
    用于在反向传播过程中限制梯度的范围
    """

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
    """
    实现向量量化，将连续向量z离散化为z_q
    :param z: [1, toksY, toksX, embed_dim], 即 [1, toksY, toksX, 256]
    :param codebook: [n_embed, embed_dim], 即 [16384, 256]
    :return: z_q, 形状与z相同
    """
    batch, toksY, toksX, embed_dim = z.shape
    n_embed, embed_dim = codebook.shape

    # 为便于操作，将z压平为z_flattened, 形状变为：[batch*toksY*toksX, embed_dim]
    z_flattened = z.reshape(-1, embed_dim)
    # z_flattened 共 batch*toksY*toksX 条向量，每条向量的纬度为 embed_dim
    # codebook    共 n_embed 条向量，每条向量的纬度为 embed_dim
    # d的形状为 [ batch*toksY*toksX , n_embed_dim ], 用于存储x_flattened和codebook各向量之间的距离
    # 其中d[i][j] 表示 z_flattened 的第i条向量与 codebook 的第j条向量之间的平方欧氏距离
    d = torch.sum(z_flattened ** 2, dim=-1, keepdim=True) + \
        torch.sum(codebook ** 2, dim=-1) - \
        2 * torch.matmul(z_flattened, codebook.T)
    # 对于x_flattened中的每条向量，寻找codebook中与其最为接近的向量，获得其在codebook中的索引，并存放与indices中
    # indices[i]存放了codebook中与x_flattened[i]最接近向量的索引
    # indices是一条纬度为batch*toksY*toksX的向量
    indices = d.argmin(-1)
    # 依据indices中的索引，从codebook中取出相应向量，构成x_q, 形状为：[batch*toksY*toksX, embed_dim]
    z_q = F.one_hot(indices, n_embed).to(d.dtype) @ codebook
    # 将x_q变形回[1, toksY, toksX, embed_dim]
    z_q = z_q.reshape_as(z)

    # d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    # indices = d.argmin(-1)
    # z_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook

    # 可以通过如下方法直接实现梯度替换，也可从代码可读性的角度出发，使用自定义的ReplaceGrad类
    # # z_q = z + (z_q - z).detach()
    # return z_q
    replace_grad = ReplaceGrad.apply
    return replace_grad(z_q, z)


class Prompt(nn.Module):
    """
    Prompt是一个模型
    Prompt存储了一个嵌入向量，该嵌入向量可以通过用户输入的提示文本得到，也可以通过用户上传的初始图片得到
    Prompt用于计算Prompt中存储的嵌入向量与图像嵌入向量之间的损失
    """

    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        # self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input, dim=-1)  # 对输入（图片嵌入向量）执行归一化操作, input_normed形状 [batch, 512]
        embed_normed = F.normalize(self.embed, dim=-1)  # 对嵌入向量执行归一化操作, embed_normed形状 [batch, 512]

        # 使用平方球面距离（Squared Great Circle Distance）作为损失函数, dist形状为 [batch, 1]
        dists = input_normed.sub(embed_normed).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()

        # 使用余弦距离作为损失函数
        # dists = 1 - torch.cosine_similarity(input_normed, embed_normed, dim=-1)

        return self.weight.abs() * dists.mean()


def parse_prompt(prompt):
    """
    解析输入的prompt
    prompt的格式为：text:weight:stop，包含三个部分，用冒号 : 分隔
    text：表示提示的文本部分。
    weight：表示权重的值，即一个浮点数，用于调整提示的影响程度。
    """
    vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size  # CLIP模型能够接受的图像分辨率
        self.cutn = cutn  # 裁剪cutn张图像，cutn即batch大小
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            # K.RandomSharpness(0.3, p=0.4),
            K.RandomHorizontalFlip(p=0.5),  # 水平翻转
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),  # 仿射变换
            K.RandomPerspective(0.2, p=0.4),  # 透视变换
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7)  # 颜色抖动
        )
        self.noise_fac = 0.1

    def forward(self, input):
        """
        对输入的图像input施加「随机裁剪」和「图像增强」形成一个批量，返回得到的batch
        :param input: [batch, channel, img_height, img_width] （实际运行时，batch为1，channel为3）
        :return: [batch, channel, cut_size, cut_size] （实际运行时，batch为超参数，channel为3）
        """

        # sideY:图像高 sideX:图像宽
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []  # cutouts中存放了施加「随机裁剪」和「图像增强」后的图像
        for _ in range(self.cutn):
            # size为图像随机裁剪大小，介于min_size和max_size之间
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())  # 设定随机裁剪的起点，X轴方向从offsetX位置开始裁剪
            offsety = torch.randint(0, sideY - size + 1, ())  # 设定随机裁剪的起点，Y轴方向从offsetY位置开始裁剪
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]  # 对原图像施加裁剪，得到裁剪后图像cutout
            # 缩放随机裁剪后图像cutout的分辨率，使其能够被CLIP模型接受，并将该图像加入cutouts中
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        # 将cutouts中的图像拼接一个形状为[cutouts_length, channel, cut_size, cut_size]的batch，并施加图像增强
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            # 添加随机噪声
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


def load_vqgan_model(config_name, checkpoint_name):
    """
    加载vqgan模型权重
    :param config_name: 配置文件名
    :param checkpoint_name: checkpoint文件名
    :return:
    """

    config_path = os.path.join('.', 'checkpoints', config_name)
    checkpoint_path = os.path.join('.', 'checkpoints', checkpoint_name)
    config = OmegaConf.load(config_path)
    # 不同配置对应的VQGAN网络架构不同
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)  # 由于VQGAN-CLIP不需要对VQGAN进行训练，因此启用评估模式
        model.init_from_ckpt(checkpoint_path)  # 从checkpoint中初始化权重
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    """
    将输入的图像调整到指定的输出尺寸
    """
    ratio = image.size[0] / image.size[1]  # 计算输入图像的宽高比
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])  # 计算输入图像和输出尺寸中较小的面积
    size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)  # 根据宽高比和面积计算调整后的尺寸
    return image.resize(size, Image.LANCZOS)  # 使用 Lanczos 插值方法调整图像尺寸并返回结果


class VQGANPainter:
    """
    VQGANPainter基于VQGAN作为生成模型实现文本生成图像（文本编辑图像）
    VQGANPainter能够生成任意分辨率的图像，但图像分辨率必须为16的倍数
    """

    def __init__(self, args):
        self.model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)  # 加载VQGAN作为生成器

        if args.discriminator == Discriminator.CLIP:
            # 加载CLIP作为判别器
            from CLIP import clip
            self.perceptor = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
        else:
            # 加载ChineseCLIP作为判别器
            import cn_clip.clip as clip
            self.perceptor = clip.load_from_name('ViT-B-16', device=device, download_root='./checkpoints/clip')[
                0].eval().requires_grad_(False).to(
                device)

        self.args = args

        cut_size = self.perceptor.visual.input_resolution  # cut_size为判别器所能接受的输入图像的分辨率
        e_dim = self.model.quantize.e_dim  # e_dim为矢量量化过程中向量的纬度
        # 解码器共有num_resolution-1个上采样层
        # z中1个token对应最终图像中1块 f*f (在当前网络中，f=16)像素大小的区域
        f = 2 ** (self.model.decoder.num_resolutions - 1)
        self.make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
        n_toks = self.model.quantize.n_e  # codebook中的向量数目（16384）
        toksX, toksY = args.size[0] // f, args.size[1] // f  # 计算隐向量z的形状
        # 生成图像的大小会向下取整为f的倍数,
        # 其中sideX为生成图像的宽，sideY为生成图像的高
        sideX, sideY = toksX * f, toksY * f
        # z_min 和 z_max 用来存储codebook中向量纬度的最小值和最大值, 其形状为[1, e_dim, 1, 1]
        # 通过计算最小值和最大值，可以获得codebook中向量权重的范围
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        if args.init_image:
            # 若提供初始图像, 则利用该图像初始化z
            pil_image = Image.open(args.init_image).convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)  # 将初始图像的大小调整为目标图像的大小
            # 利用VQGAN的编码器将初始图像编码为隐变量
            self.z, *_ = self.model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
        else:
            # 反之则从codebook中随机抽取向量，对隐变量z进行初始化
            # 先随机生成形状为 [toksY * toksX] 的随机整数张量，取值范围为 [0, n_toks), 再生成对应的one_hot向量， 形状为 [toksY * toksX, n_toks]
            one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
            self.z = one_hot @ self.model.quantize.embedding.weight
            self.z = self.z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)  # 隐变量z是网络需要优化的目标，因此需要梯度
        self.opt = optim.Adam([self.z], lr=args.step_size)  # 使用Adam优化器更新z

        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

        self.pMs = []  # pMs用于管理所有prompt

        # 解析并处理输入的所有文本prompt
        for prompt in args.prompts:
            txt, weight = parse_prompt(prompt)
            # 首先使用 tokenize(txt) 将文本转换为 CLIP 模型可接受的标记化表示
            # 随后调用 encode_text 将标记化后的文本编码为文本嵌入向量
            embed = self.perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            # 构建对应的prompt模型，并加入pMs中
            self.pMs.append(Prompt(embed, weight).to(device))

        # 解析并处理输入的所有图片prompt
        for prompt in args.image_prompts:
            path, weight = parse_prompt(prompt)  # 解析图像路径和权重
            # 对图像进行调整大小，使其符合指定的目标大小 (sideX, sideY)
            img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
            batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
            embed = self.perceptor.encode_image(self.normalize(batch)).float()
            self.pMs.append(Prompt(embed, weight).to(device))

        # 处理噪声prompt
        for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)  # 创建一个随机数生成器对象 gen，并使用指定的种子 seed 进行设置。
            # 创建一个形状为[1, output_dim] 的空张量embed，并使用正态分布进行填充
            embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(generator=gen)
            self.pMs.append(Prompt(embed, weight).to(device))

    def synth(self, z):
        """
        根据隐变量z生成图像
        """
        clamp_with_grad = ClampWithGrad.apply

        # 对z进行矢量量化操作，得到z_q (在矢量量化前后，需要分别改变z和z_q的形状)
        z_q = vector_quantize(z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        # 通过解码器生成图像，并对生成的图像做反归一化操作，同时利用梯度裁剪，将确保图像像素值在 [0, 1] 的范围内，返回最终生成的图像
        return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    def checkin(self, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        tqdm.write(f'i: {cur_iter}, loss: {sum(losses).item():g}, losses: {losses_str}')
        out = self.synth(self.z)
        TF.to_pil_image(out[0].cpu()).save('progress.png')
        # todo 显示图片

    def ascend_txt(self):
        """
        ascend_txt不断更新隐变量z，以便更好地满足文本的要求
        """

        out = self.synth(self.z)  # out为生成图像
        # iii为图像嵌入向量，通过判别器对「随机裁剪+图像增强」后的图像编码得到
        image_embed = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()

        result = []

        # if self.args.init_weight:
        #     result.append(F.mse_loss(self.z, self.z_orig) * self.args.init_weight / 2)

        for prompt in self.pMs:
            # 将图像编码向量image_embed作为输入，计算iii关于每个prompt的损失，并将其添加到结果列表中
            result.append(prompt(image_embed))

        # 保存生成的图片，作为中间结果，用于在网页前端动态显示以及生成视频
        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:, :, :]
        img = np.transpose(img, (1, 2, 0))
        filename = f"static/steps/{cur_iter + 1:04}.png"
        imageio.imwrite(filename, np.array(img))

        return result

    def train_once(self):
        """
        进行一次训练
        """
        self.opt.zero_grad()  # 将优化器的梯度清零

        lossAll = self.ascend_txt()
        loss = sum(lossAll)
        loss.backward()
        self.opt.step()  # 更新模型参数

        # print(f'{cur_iter}: total= {loss.item():.1f} ')

        with torch.no_grad():
            # 对生成的图像张量 self.z 进行裁剪，确保像素值在一定范围内
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

    def train(self, threadCTL):
        """
        推理（也是训练），实现文本生成图像
        :param threadCTL: 线程控制块
            threadCTL['interrupt']  是否需要终止任务
            threadCTL['lastIter']   模型最终迭代次数
        """

        global cur_iter
        # cur_iter表示当前迭代的次数，cur_iter.png是当前已生成的最新图片
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
