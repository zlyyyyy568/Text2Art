import argparse
import os
import shutil

import numpy as np
from PIL import Image

import torch
from VQGANPainter import VQGANPainter
from BigGANPainter import BigGANPainter
from VQGANPainterWithMSE import VQGANPainterWithMSE
from tqdm import tqdm

from enum import Enum

from models import ImageType, AspectRatio, Generator, Discriminator
from MyQueue import MyQueue


class TaskStatus(Enum):
    """
    任务控制块的状态
    """

    WAIT = 0  # 处于等待队列
    RUNNING = 1  # 正在运行
    SUCCESS = 2  # 成功运行完毕
    INTERRUPT = 3  # 运行中断
    ERROR = 4  # 运行错误


class TCB:
    """
    任务控制块
    """

    def __init__(self, user_id, prompts: str, image_type: ImageType, iter_time,
                 aspect_ratio: AspectRatio, width, height, generator: Generator, discriminator: Discriminator,
                 initial_image: str, target_images: str):
        self.user_id = user_id

        self.prompts = prompts
        self.image_type = image_type
        self.iter_time = iter_time
        self.aspect_ratio = aspect_ratio
        self.width = width
        self.height = height
        self.generator = generator
        self.discriminator = discriminator
        self.taskStatus = TaskStatus.WAIT

        self.tempResultDir = os.path.join('.', 'static/temp/' + str(user_id))
        self.tempImageName = 'temp.png'
        self.tempVideoName = 'temp.mp4'

        self.initialImage = ''
        if initial_image != '':
            self.initialImage = 'initialImage' + os.path.splitext(initial_image)[1]
            self.initialImage = os.path.join(self.tempResultDir, self.initialImage)

        self.targetImage = ''
        if target_images != '':
            self.targetImage = 'targetImage' + os.path.splitext(target_images)[1]
            self.targetImage = os.path.join(self.tempResultDir, self.targetImage)

    def __eq__(self, other):
        if isinstance(other, TCB):
            return self.user_id == other.user_id
        elif isinstance(other, int):
            return self.user_id == other
        return False

    def __str__(self):
        return f"user_id <{self.user_id}> "


class JinJaTCB:
    """
    将tcb转换为前端Jinja2模版中能够使用的数据结构
    """

    def __init__(self, tcb: "TCB" = None):
        if tcb is not None:
            self.user_id = tcb.user_id

            self.prompts = tcb.prompts
            self.image_type = tcb.image_type.name
            self.iter_time = tcb.iter_time
            self.aspect_ratio = tcb.aspect_ratio.name
            self.width = tcb.width
            self.height = tcb.height
            self.generator = tcb.generator.name
            self.discriminator = tcb.discriminator.name
            self.taskStatus = tcb.taskStatus

            self.imgSrc = f"/static/temp/{self.user_id}/temp.png"
            self.videoSrc = f"/static/temp/{self.user_id}/temp.mp4"
            print('----')
            print(self.videoSrc)
            print('----')
        else:
            self.user_id = -1

            self.prompts = ""
            self.image_type = ""
            self.iter_time = 700
            self.aspect_ratio = AspectRatio.CUSTOMIZE.name
            self.width = 512
            self.height = 512
            self.generator = Generator.VQGAN.name
            self.discriminator = Discriminator.CLIP.name

            self.imgSrc = f"/static/resource/default-image-square.png"


class AIPainter:
    """
    AIPainter对上提供一层抽象，通过线程控制块与后端交互，实现文本生成图像，并利用生成的中间图像制作视频
    在内部则根据tcb中的配置，实现不同的组合式生成对抗网络

    生成的图像默认保存为 static/temp/user_id/temp.png
    生成的视频默认保存为 static/temp/user_id/temp.mp4
    """

    def __init__(self, tcb: TCB, finishQueue: MyQueue):
        self.finishQueue = finishQueue
        self.tcb = tcb
        if self.tcb.generator == Generator.VQGAN:
            # 使用VQGAN作为判别器
            self.args = self.constructVQGAN(hasRegularization=False)
            self.painter = VQGANPainter(self.args)
        elif self.tcb.generator == Generator.VQGANMSE:
            # 使用带正则项损失的VQGAN作为判别器
            self.args = self.constructVQGAN(hasRegularization=True)
            self.painter = VQGANPainterWithMSE(self.args)
        else:
            # 使用BigGAN作为判别器
            self.args = self.constructBigGAN()
            self.painter = BigGANPainter(self.args)

        # 线程控制块钱
        self.threadCTL = {
            'imageGenerated': False,  # 完成图像生成 或 图像生成终止
            'interrupt': False,  # 终止图像生成
            'videoGenerated': False,  # 完成视频生成
            'finish': False,  # 线程任务结束（ 生成终止 或 完成图像、视频的生成)
            'lastIter': 0,  # 图像的最后迭代次数
        }

    def run(self):
        # 生成图像
        self.painter.train(self.threadCTL)
        self.threadCTL['imageGenerated'] = True

        if self.threadCTL['interrupt']:
            # 若线程中断运行
            self.threadCTL['finish'] = True
            self.tcb.taskStatus = TaskStatus.INTERRUPT
            print('图像生成终止!!!')
        else:
            # 完成图像生成任务，保存临时图像
            # 先检查用户目录是否存在，若不存在，则创建
            if not os.path.exists(self.tcb.tempResultDir):
                os.mkdir(self.tcb.tempResultDir)
            srcImage = os.path.join('static/steps', f"{self.getCurIter():04d}.png")
            destImage = os.path.join(self.tcb.tempResultDir, self.tcb.tempImageName)
            shutil.copy(srcImage, destImage)

            # 开始生成视频
            # 保存临时视频的功能在 generate_video 中实现
            self.generate_video(self.threadCTL['lastIter'])
            self.threadCTL['videoGenerated'] = True
            self.threadCTL['finish'] = True

            # 将tcb加入完成队列
            self.finishQueue.enqueue(self.tcb)
            self.tcb.taskStatus = TaskStatus.SUCCESS

    def stop(self):
        """
        AIPainter运行中断
        """

        self.threadCTL['interrupt'] = True

    def getCurIter(self):
        return self.painter.getCurIter()

    def generate_video(self, last_frame):
        """
        根据模型运行过程中生成的中间结果，生成相应视频
        :param 模型生成的最后一帧图片
        """

        init_frame = 1  # 视频开始的帧数
        last_frame += 1  # 视频最后一帧的帧数(不包括last_frame，故需要加1)

        total_frames = last_frame - init_frame  # 视频的总帧数

        min_fps = 10  # 最小帧率
        max_fps = 30  # 最大帧率

        length = 15  # 期望的视频播放时长（秒）

        frames = []  # 视频的总帧数
        tqdm.write('Generating video...')
        for i in range(init_frame, last_frame):
            # 遍历从 init_frame 到 last_frame 的帧数，读取对应的图像文件并将其添加到 frames 列表中。
            filename = f"static/steps/{i:04}.png"
            frames.append(Image.open(filename))

        # 根据总帧数和期望的视频时长计算得到的帧率，限制在 min_fps 和 max_fps 之间。
        fps = np.clip(total_frames / length, min_fps, max_fps)

        # 设置输出视频的文件路径和名称
        video_filename = os.path.join(self.tcb.tempResultDir, self.tcb.tempVideoName)
        print(video_filename)

        from subprocess import Popen, PIPE
        # 使用 subprocess.Popen 创建一个子进程在终端执行FFmpeg命令
        # 利用FFmpeg工具，将生成的图像帧逐帧输入，并通过指定的参数将这些帧编码为视频文件

        # -y：在不提示用户的情况下覆盖输出文件（如果它已经存在）。
        # -f image2pipe：将图像作为输入流处理。
        # -vcodec png：将输入图像编码为PNG格式。
        # -r：设置视频帧率为fps。
        # -i -：从标准输入（stdin）读取图像流。
        # -vcodec libx264：使用libx264编码器将输入视频流编码为H.264视频格式。
        # -r：设置输出视频的帧率为fps。
        # -pix_fmt yuv420p：设置输出视频的像素格式为yuv420p。
        # -crf 17：设置输出视频的质量。
        # -preset veryslow：设置编码速度为非常慢。
        #  video_filename：输出视频文件的路径和文件名。
        p = Popen(
            ['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps), '-i', '-', '-vcodec', 'libx264',
             '-r', str(fps), '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'veryslow', video_filename], stdin=PIPE)
        for im in tqdm(frames):
            im.save(p.stdin, 'PNG')  # 将每一帧图像保存到子进程的标准输入流中
        p.stdin.close()  # 关闭子进程的标准输入流

        print("Compressing video...")
        p.wait()  # 等待子进程执行完毕
        print("Video ready")

    def constructVQGAN(self, hasRegularization):
        """
        根据tcb生成VQGAN的网络配置
        :param hasRegularization: 是否需要正则项
        :return: 网络配置
        """

        # 设置选用的vqgan模型为vqgan_imagenet_f16_16384
        # ["vqgan_imagenet_f16_16384", "vqgan_imagenet_f16_1024", "wikiart_16384", "coco", "faceshq", "sflckr"]
        model_selected = "vqgan_imagenet_f16_16384"
        initial_image = self.tcb.initialImage  # 设置初始图像
        target_images = self.tcb.targetImage  # 设置目标图像
        seed = -1  # 随机种子，用于控制生成图像的随机性 若为-1，则表示不使用随机种子。
        input_images = ""

        # model_names = {"vqgan_imagenet_f16_16384": 'ImageNet 16384', "vqgan_imagenet_f16_1024": "ImageNet 1024",
        #                "wikiart_16384": "WikiArt 16384", "coco": "COCO-Stuff", "faceshq": "FacesHQ",
        #                "sflckr": "S-FLCKR"}
        # model_name = model_names[model_selected]

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

        prompts = [frase.strip() for frase in self.tcb.prompts.split("|")]  # 解析用户输入的多个文本提示，将其分为多个prompt
        if prompts == ['']:
            prompts = []

        promptKeywords = self.tcb.image_type
        if promptKeywords is not None:
            # 将用户选择的图像风格转为对应的prompt
            if promptKeywords == ImageType.ARTSTATION:
                prompts.append('Trending on Artstation')
            elif promptKeywords == ImageType.UNREAL_ENGINE:
                prompts.append('Unreal Engine')
            elif promptKeywords == ImageType.ANIME:
                prompts.append('Anime')
            elif promptKeywords == ImageType.MINIMALIST:
                prompts.append("Minimalist")
            else:
                # 如果promptKeyword=="Normal"，不做任何处理
                pass

        if not hasRegularization:
            # 使用原始的 VQGAN
            args = argparse.Namespace(
                prompts=prompts,
                image_prompts=target_images,
                noise_prompt_seeds=[],
                noise_prompt_weights=[],
                size=[self.tcb.width, self.tcb.height],
                init_image=initial_image,
                init_weight=0.,

                discriminator=self.tcb.discriminator,
                clip_model='ViT-B/32',
                # clip_model='ViT-L/14',
                vqgan_config=f'{model_selected}.yaml',
                vqgan_checkpoint=f'{model_selected}.ckpt',
                step_size=0.1,
                cutn=64,
                cut_pow=1.,
                display_freq=50,
                seed=seed,
                max_iterations=self.tcb.iter_time,
            )
        else:
            # 使用带正则化损失的VQGAN
            args = argparse.Namespace(
                prompts=prompts,
                image_prompts=target_images,
                noise_prompt_seeds=[],
                noise_prompt_weights=[],
                size=[self.tcb.width, self.tcb.height],
                init_image=initial_image,
                init_weight=1.5,  # 正则项的初始权重
                # init_weight=2.,

                discriminator=self.tcb.discriminator,
                clip_model='ViT-B/32',
                vqgan_config=f'{model_selected}.yaml',
                vqgan_checkpoint=f'{model_selected}.ckpt',

                step_size=0.95,
                cutn=64,
                cut_pow=1.,
                display_freq=50,
                seed=seed,
                max_iterations=self.tcb.iter_time,

                # 使用L2正则化，即均方误差MSE
                mse_withzeros=True,  # 使用全零向量计算MSE
                mse_decay_rate=50,  # 正则化权重的衰减速率，即模型每迭代mse_decay_rate，便衰减一次权重
                mse_epoches=5,  # 正则项权重共需衰减多少次。其中模型每迭代mse_decay_rate，便算作一个epoch，衰减一次权重
            )

        args.cutn = 64

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

        return args

    def constructBigGAN(self):
        """
        根据tcb生成BigGAN的网络配置
        """
        args = argparse.Namespace(
            prompts=self.tcb.prompts,
            # ['From prompt', 'Random class', 'Random Dirichlet', 'Random mix'] {allow-input: true}
            initial_class='Random mix',
            optimize_class=True,
            class_smoothing=0.1,
            truncation=1,
            init_weight=0.,

            discriminator=self.tcb.discriminator,
            clip_model='ViT-B/32',
            augmentations=64,
            learning_rate=0.1,
            class_ent_reg=0.0001,  # 类向量熵的正则化程度
            max_iterations=self.tcb.iter_time,
        )

        return args


if __name__ == '__main__':
    # prompts = "a boy with a dog in the sunshine"  # @param {type:"string"}
    # prompts = "A boy is running in the rain"  # @param {type:"string"}
    pass
