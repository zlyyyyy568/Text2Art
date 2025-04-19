import argparse

import torch
import torchvision
import numpy as np
import imageio
from scipy.stats import truncnorm, dirichlet
from pytorch_pretrained_biggan import convert_to_images, one_hot_from_names, BigGAN
from tqdm import tqdm

from models import Discriminator
from VQGANPainter import resample

cur_iter = 0


def saveImage(out, name):
    """
    保存图像
    :param out: tensor张量
    :param name: 图像的命名
    :return:
    """
    with torch.no_grad():
        al = out.cpu().numpy()
    img = convert_to_images(al)[0]
    imageio.imwrite(name, np.asarray(img))


def checkin(total_loss, loss, reg, values, out):
    global sample_num
    name = f'./BigSleepResult/{sample_num:04}.jpg'
    saveImage(out, name)
    # todo show image
    # display(Image(name))
    print('%d: total=%.1f cos=%.1f reg=%.1f components: >=0.5=%d, >=0.3=%d, >=0.1=%d\n' % (
        sample_num, total_loss, loss, reg, np.sum(values >= 0.5), np.sum(values >= 0.3), np.sum(values >= 0.1)))
    sample_num += 1


class BigGANPainter:
    """
    使用BigGAN作为生成器，实现文本生成图像
    """

    def __init__(self, args):
        # 加载BigGAN的权重
        self.model = BigGAN.from_pretrained('biggan-deep-512').cuda().eval()

        if args.discriminator == Discriminator.CLIP:
            # 加载CLIP作为判别器
            from CLIP import clip
            self.perceptor = clip.load(args.clip_model, device='cuda')[0].eval().requires_grad_(False)
        else:
            # 加载ChineseCLIP作为判别器
            import cn_clip.clip as clip
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.perceptor = clip.load_from_name('ViT-B-16', device=device, download_root='./checkpoints/clip')[
                0].eval().requires_grad_(False).to(
                device)

        self.args = args

        im_shape = [512, 512, 3]  # 所选用的BigGAN模型只能生成512*512分辨率的图像
        self.sideX, self.sideY, self.channels = im_shape

        seed = None
        state = None if seed is None else np.random.RandomState(seed)
        np.random.seed(seed)

        # 生成一个大小为[1, 128]的随机噪声，该噪声服从截断正态分布
        # 截断范围为[-2 * truncation, 2 * args.truncation]
        # 对于BigGAN，截断区间的越窄，生成图像的质量会越高，但多样性将下降
        self.noise_vector = truncnorm.rvs(-2 * args.truncation, 2 * args.truncation, size=(1, 128),
                                          random_state=state).astype(np.float32)

        # 初始化类向量
        if args.initial_class.lower() == 'random class':
            # 在类向量中随机选择一个类别，并使该类别的概率较高，其他类别的概率较低
            self.class_vector = np.ones(shape=(1, 1000), dtype=np.float32) * args.class_smoothing / 999
            self.class_vector[0, np.random.randint(1000)] = 1 - args.class_smoothing
        elif args.initial_class.lower() == 'random mix':
            # 各种类的随机混合
            self.class_vector = np.random.rand(1, 1000).astype(np.float32)
        self.eps = 1e-8
        self.class_vector = np.log(self.class_vector + self.eps)

        self.noise_vector = torch.tensor(self.noise_vector, requires_grad=True, device='cuda')
        self.class_vector = torch.tensor(self.class_vector, requires_grad=True, device='cuda')

        params = [self.noise_vector]
        if args.optimize_class:
            # 当optimize_class为True时，反向传播时对类向量进行更新
            params += [self.class_vector]
        self.optimizer = torch.optim.Adam(params, lr=args.learning_rate)  # 使用Adam算法更新噪声向量和类向量

        # 通过CLIP将提示文本编码为文本嵌入量
        self.txt_embed = self.perceptor.encode_text(clip.tokenize(args.prompts).to('cuda')).float()

        self.cut_size = self.perceptor.visual.input_resolution  # 解码器能够接受的图像分辨率

        self.nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                    (0.26862954, 0.26130258, 0.27577711))

    def ascend_txt(self):
        global cur_iter

        # 通过使用clamp函数对噪声项链进行截断，将其限制在 [-2 * self.args.truncation, 2 * self.args.truncation] 的范围内
        noise_vector_trunc = self.noise_vector.clamp(-2 * self.args.truncation, 2 * self.args.truncation)
        # 对class_vector进行归一化操作，将其转换为概率分布，类向量中的每一维度表示相应类别的概率
        class_vector_norm = torch.nn.functional.softmax(self.class_vector)
        out = self.model(noise_vector_trunc, class_vector_norm, self.args.truncation)

        # 保存生成的图片
        saveImage(out, f'static/steps/{cur_iter + 1:04d}.png')

        max_size = min(self.sideX, self.sideY)
        min_size = min(self.sideX, self.sideY, self.cut_size)
        cutouts = []  # cutouts中存放了施加「随机裁剪」后的图像
        for _ in range(self.args.augmentations):
            # size为图像随机裁剪大小，介于min_size和max_size之间
            size = int(torch.rand([]) * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, self.sideX - size + 1, ())  # 设定随机裁剪的起点，X轴方向从offsetX位置开始裁剪
            offsety = torch.randint(0, self.sideY - size + 1, ())  # 设定随机裁剪的起点，Y轴方向从offsetY位置开始裁剪
            cutout = out[:, :, offsety:offsety + size, offsetx:offsetx + size]  # 对原图像施加裁剪，得到裁剪后图像cutout
            # 缩放随机裁剪后图像cutout的分辨率，使其能够被CLIP模型接受，并将该图像加入cutouts中
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        # 将cutouts中的图像拼接一个形状为[cutouts_length, channel, cut_size, cut_size]的batch，并施加图像增强
        batch = torch.cat(cutouts, dim=0)

        # 将图片编码为图片嵌入向量
        image_embed = self.perceptor.encode_image(batch)

        # 使用余弦距离作为损失函数
        factor = 100
        loss = factor * (1 - torch.cosine_similarity(image_embed, self.txt_embed, dim=-1).mean())

        # 使用平方球面距离（Squared Great Circle Distance）作为损失函数, dist形状为 [batch, 1]
        # image_norm = image_embed / image_embed.norm(dim=-1, keepdim=True)
        # txt_norm = self.txt_embed / self.txt_embed.norm(dim=-1, keepdim=True)
        # loss = image_norm.sub(txt_norm).norm(dim=-1).div(2).arcsin().pow(2).mul(2).mean()

        total_loss = loss
        reg = torch.tensor(0., requires_grad=True)
        if self.args.optimize_class and self.args.class_ent_reg:
            # 在图像迭代的过程中对类向量进行约束，使类向量中某些特定类的概率较高，最终生成的图像倾向于某些特定类别
            # 该约束通过向损失中添加正则项实现
            # reg是类向量熵的正则化项，通过减小类向量的熵从而实现类向量概率的集中
            reg = -factor * self.args.class_ent_reg * \
                  (class_vector_norm * torch.log(class_vector_norm + self.eps)).sum()
            total_loss += reg

        # 打印损失
        values = class_vector_norm.cpu().detach().numpy()
        print(
            f'{cur_iter}: total= {total_loss.item():.1f} ' +
            f'cos= {loss.item():.1f} reg={reg.item():.1f} ' +
            f'components: >=0.5 = {np.sum(values >= 0.5):d}, >=0.3= {np.sum(values >= 0.3):d}, >=0.1= {np.sum(values >= 0.1):d}\n'
        )

        return total_loss

    def train(self, threadCTL):
        global cur_iter
        # cur_iter表示当前迭代的次数，也可以表示当前已生成的图片
        cur_iter = 0
        try:
            with tqdm(self.args.max_iterations) as pbar:
                isInterrupt = threadCTL['interrupt']
                while not isInterrupt:
                    self.optimizer.zero_grad()  # 将优化器的梯度清零

                    loss = self.ascend_txt()
                    loss.backward()
                    self.optimizer.step()  # 更新模型参数

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
    args = argparse.Namespace(
        prompts='summer night',
        initial_class='Random mix',
        # @param ['From prompt', 'Random class', 'Random Dirichlet', 'Random mix'] {allow-input: true}
        optimize_class=True,  # @param {type:'boolean'}
        class_smoothing=0.1,  # @param {type:'number'}
        truncation=1,  # @param {type:'number'}
        init_weight=0.,
        clip_model='ViT-B/32',
        augmentations=64,  # @param {type:'integer'}
        learning_rate=0.1,
        class_ent_reg=0.0001,  # @param {type:'number'}
        max_iterations=100,
    )

    painter = BigGANPainter(args)
    threadCTL = {}
    threadCTL['interrupt'] = False
    painter.train(threadCTL)
