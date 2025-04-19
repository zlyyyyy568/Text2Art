from datetime import datetime

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from enum import Enum

app = Flask(__name__)

# 2.设置数据库的配置信息
# 设置数据库的链接信息,
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:123456@127.0.0.1:3306/text2art"
# 该字段增加了大量的开销,会被禁用,建议设置为False
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# 3.创建sqlalchemy对象db,关联app
db = SQLAlchemy(app)


class User(db.Model):
    # __tablename__ = "students"
    # 主键, 参数1: 表示id的类型, 参数2: 表示id的约束类型
    id = db.Column(db.Integer, primary_key=True)
    account = db.Column(db.String(32), unique=True)
    password = db.Column(db.String(32))


class ImageType(Enum):
    NORMAL = 'Normal'
    ARTSTATION = 'Artstation'
    UNREAL_ENGINE = 'Unreal Engine'
    ANIME = 'Anime'
    MINIMALIST = 'Minimalist'


class Generator(Enum):
    VQGAN = 'VQGAN'
    VQGANMSE = 'VQGAN-MSE'
    BIGGAN = 'BigGAN'


class Discriminator(Enum):
    CLIP = 'CLIP'
    CHINESE_CLIP = 'Chinese-CLIP'


class AspectRatio(Enum):
    CUSTOMIZE = 'customize'
    WIDESCREEN = 'widescreen'
    PORTRAIT = 'portrait'
    SQUARE = 'square'


class Image(db.Model):
    # __tablename__ = "students"
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

    user_id = db.Column(db.Integer, db.ForeignKey(User.id))


class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(256))
    image_id = db.Column(db.Integer, db.ForeignKey(Image.id))
    user_id = db.Column(db.Integer, db.ForeignKey(User.id))
    time = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()
    # db.session.add(Comment())
    # db.session.commit()
    # print(Comment.query.all()[0].time.strftime('%Y年%m月%d日 %H:%M'))
