*以下内容使用Typora编写，用其他MarkDown编辑器打开可能会存在格式错误的问题*

该文件夹中存放了毕设过程中的所有代码，可用于运行一个名为Text2Art的网站。

Text2Art需要GPU才能够正常运行，如果没有提供GPU，则Text2Art无法实现文本生成图像任务，只能用作简单的浏览。

Text2Art的前端采用HTML、CSS和JavaScript编写前端页面（使用了Flask的JinJa2模版，并且部分代码用到了BootStrap框架），后端采用Flask框架，使用sqlalchemy操作数据库，数据库使用的是MySQL，所有代码均在PyCharm中编写。

在命令行中，通过`python app.py`即可运行Text2Art  

现对代码文件进行说明

- app.py：Text2art的入口文件
- 模型相关代码
  - AIPainter.py：对上提供接口，Text2Art通过AIPainter.py运行模型
  - VQGANPainter.py：实现VQGAN-CLIP和VQGAN-ChineseCLIP
  - VQGANPainterWIthMSE.py：与VQGANPainter几乎相同，不同之处在于VQGANPainterWithMSE 中的损失函数额外增加了一个正则项
  - BigGANPainter.py：实现BigGAN-CLIP和BigGAN-ChineseCLIP
  - cloab.py：可直接运行的VQGAN-CLIP模型，用于debug
  - CLIPBigGAN.py：可直接运行的BigGAN-CLIP模型，用于debug
- 系统相关代码
  - 后端代码
    - index.py：**首页**页面对应的后端代码
    - painter.py：**绘制**页面对应的后端代码
    - gallery.py：**画廊**页面对应的后端代码
    - homepage.py：**我的**页面对应的后端代码
  - MyQueue.py：实现了一个多线程安全的队列，用于用户请求调度
  - models.py：用于创建数据库的模型类
- 杂项
  - createDatabase.py：由于代码存在未解决的Bug，所以需要将model.py中的相关代码复制到createDatabase.py中，通过调用`db.create_all()`创建对应的数据库表。
  - imageProcess.py：通过TinyPNG提供的API实现图像压缩，成功将Word从150MB压缩至20MB，简直就是神迹！



Text2Art的运行环境已被导出为「**environment.yml**」文件，可以通过conda恢复

下图为Text2Art所使用的数据库界面，数据库中的内容已被导出为SQL语句，放在「**重建数据库**」文件夹中。

![Navicat中的数据库界面](/Users/Maple/text2art/Navicat中的数据库界面.png)

