from datetime import datetime

from app import db
from enum import Enum


class ImageType(Enum):
    """
    图片风格
    """

    NORMAL = 'Normal'
    ARTSTATION = 'Artstation'
    UNREAL_ENGINE = 'Unreal Engine'
    ANIME = 'Anime'
    MINIMALIST = 'Minimalist'


class Generator(Enum):
    """
    生成器类型
    """

    VQGAN = 'VQGAN'
    VQGANMSE = 'VQGAN-MSE'
    BIGGAN = 'BigGAN'


class Discriminator(Enum):
    """
    判别器类型
    """

    CLIP = 'CLIP'
    CHINESE_CLIP = 'Chinese-CLIP'


class AspectRatio(Enum):
    """
    图像长宽比
    """

    CUSTOMIZE = 'customize'
    WIDESCREEN = 'widescreen'
    PORTRAIT = 'portrait'
    SQUARE = 'square'


class User(db.Model):
    """
    学生表
    """

    # __tablename__ = "students"
    # 主键, 参数1: 表示id的类型, 参数2: 表示id的约束类型
    # 用户ID
    id = db.Column(db.Integer, primary_key=True)
    # 用户账户名
    account = db.Column(db.String(32), unique=True)
    # 用户密码
    password = db.Column(db.String(32))


class Image(db.Model):
    """
    图片表
    """

    # 主键, 参数1: 表示id的类型, 参数2: 表示id的约束类型
    # 图片 id
    id = db.Column(db.Integer, primary_key=True)
    # 图片名
    name = db.Column(db.String(256))
    # 图像类型 Normal, Artstation, Unreal Engine, Anime, Minimalist
    image_type = db.Column(db.Enum(ImageType), default=ImageType.NORMAL)
    # 生成模型 VQGAN, BigGAN
    generator = db.Column(db.Enum(Generator), default=Generator.VQGAN)
    # 判别模型 CLIP, ChineseCLIP
    discriminator = db.Column(db.Enum(Discriminator), default=Discriminator.CLIP)

    # 图像的长宽比，即宽度与高度之间的比值 customize, widescreen, portrait, square
    aspect_ratio = db.Column(db.Enum(AspectRatio), default=AspectRatio.CUSTOMIZE)
    # 图像的宽度
    width = db.Column(db.SmallInteger, default=0)
    # 图像的高度
    height = db.Column(db.SmallInteger, default=0)
    # 图像的迭代次数
    iter_time = db.Column(db.SmallInteger, default=0)

    # 其他标签
    tag = db.Column(db.String(256), default="")

    # 外键，用户ID
    user_id = db.Column(db.Integer, db.ForeignKey(User.id))

    # 图片创建时间
    time = db.Column(db.DateTime, default=datetime.utcnow)
    # 图片喜欢数
    like = db.Column(db.SmallInteger, default=0)


class Comment(db.Model):
    """
    评论表
    """

    # 评论ID
    id = db.Column(db.Integer, primary_key=True)
    # 评论内容
    text = db.Column(db.String(256))
    # 外键，图片ID，即评论是关于那张图片的
    image_id = db.Column(db.Integer, db.ForeignKey(Image.id))
    # 外溅，用户ID，即评论是哪位用户发布的
    user_id = db.Column(db.Integer, db.ForeignKey(User.id))
    # 评论发布时间
    time = db.Column(db.DateTime, default=datetime.utcnow)
