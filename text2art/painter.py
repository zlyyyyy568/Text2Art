import argparse
import base64
import json
import os
import shutil
import threading
import time

from flask import Blueprint, request, render_template, make_response, jsonify, session
from werkzeug.utils import secure_filename

from AIPainter import AIPainter, TCB, TaskStatus, JinJaTCB

blue_painter = Blueprint("blue-painter", __name__)

from app import db, app
from models import User, Image
from models import ImageType, Generator, Discriminator, AspectRatio

from constraint import imagesDirectory, videosDirectory
from utils import checkIsLogin
from enum import Enum
from MyQueue import MyQueue

painter: AIPainter = None
protect_thread = None

waitingQueue = MyQueue()
finishedQueue = MyQueue()


def protectThread():
    """
    守护线程
    """

    global painter
    while True:
        if painter is not None:
            # 若当前有正在运行的任务 或 已完成任务但painter未被回收
            threadCTL = painter.threadCTL
            if threadCTL['finish'] and not threadCTL['interrupt']:
                # 图像完成视频生成任务
                print('protect thread set painter to None')
                painter = None
        elif (painter is None) and (not waitingQueue.isEmpty()):
            # 若当前没有正在运行的任务，且等待队列中有用户提交的任务

            # 由于painter初始化需要时间
            # 如果将tcb直接从队头取出，在painter初始化的这段时间内会出现异常（tcb即不处于运行状态，又不在等待队列中）
            tcb = waitingQueue.peek()

            print('\n' + '-' * 10)
            print(f"运行用户< {tcb.user_id} > 提交的任务")
            print(tcb.prompts)
            print('-' * 10 + '\n')
            tcb.taskStatus = TaskStatus.RUNNING  # 将该tcb的状态设置为运行态
            painter = AIPainter(tcb, finishedQueue)  # 初始化模型

            waitingQueue.dequeue()  # 将该tcb从等待队列中移除
            painter_thread = threading.Thread(target=painter.run, name="painterThread")

            painter_thread.start()

        time.sleep(2)


class UserStatus(Enum):
    """
    用户的状态
    """
    NO_TASK = 0  # 当前用户没有任何任务
    HAS_WAIT_TASK = 1  # 用户的任务在等待队列中
    HAS_RUNNING_TASK = 2  # 用户的任务正在运行
    HAS_FINISH_TASK = 3  # 用户的任务已经完成，等待确认


def getUserInfo(user_id):
    """
    获取用户的状态和用户的TCB
    :param user_id: 用户的ID
    :return: (用户状态, 用户的TCB)
    """

    # 在已完成队列中查找
    for tcb in finishedQueue:
        if tcb.user_id == user_id:
            return UserStatus.HAS_FINISH_TASK, tcb

    # 查看是否为当前正在运行的任务
    if painter is not None and user_id == painter.tcb.user_id:
        return UserStatus.HAS_RUNNING_TASK, painter.tcb

    # 最后查看是否在等待队列中
    for tcb in waitingQueue:
        if tcb.user_id == user_id:
            return UserStatus.HAS_WAIT_TASK, tcb

    return UserStatus.NO_TASK, None


# @blue_painter.route('/')
@blue_painter.route('/painter', methods=['POST', 'GET'])
def painter_view():
    global protect_thread
    if protect_thread is None:
        # 当第一次进入页面时，启动守护线程
        protect_thread = threading.Thread(target=protectThread, name="protectThread")
        protect_thread.start()
        print('- - - 启动守护进程!')

    if request.method == "GET":
        # GET请求用于返回绘制页面
        # 根据用户是否登录，返回不同页面
        if checkIsLogin():
            user_id = session['user_id']
            username = session['username']

            user_status, user_tcb = getUserInfo(user_id)

            # 根据用户当前状态，返回不同页面
            if user_status == UserStatus.NO_TASK:
                # 用户当前没有任务
                return render_template('painter_UI.html', username=username, user_id=user_id,
                                       user_status=user_status.name,
                                       user_tcb=JinJaTCB())
            else:
                # 若用户有提交的任务，则将该tcb取出，转化为前端可以接受的JinJaTCB传给前端
                # 由前端更具JinJaTCB对代码逻辑做进一步的处理
                return render_template('painter_UI.html', username=username, user_id=user_id,
                                       user_status=user_status.name,
                                       user_tcb=JinJaTCB(user_tcb))

        return render_template('painter_UI.html', user_tcb=JinJaTCB())
    else:
        # POST请求用于处理用户在绘制页面提交的绘制请求
        global painter

        # 打印等待队列、正在运行的tcb和完成队列
        print('\n' + '-' * 10)
        print('wait queue: ', end="")
        waitingQueue.showList()
        print('painter: ', end="")
        print(painter)
        print('finish list: ', end="")
        finishedQueue.showList()
        print('-' * 10 + '\n')

        # 读取用户提价的表单信息
        generator = request.form.get('generatorRadios')  # 生成模型
        discriminator = request.form.get('discriminatorRadios')  # 判别模型

        prompts = request.form.get("prompt")  # 文本提示
        max_iterations = int(request.form.get('max-iterations'))  # 最大迭代次数
        width = int(request.form.get('width-input'))  # 生成图像的宽度
        height = int(request.form.get('height-input'))  # 生成图像的高度

        image_type = request.form.get('promptKeywords')  # 图像风格
        if image_type is None:
            image_type = "Normal"
        aspect_ratio = request.form.get('AspectRatioSelect')  # 图像长宽比

        initial_image = request.form.get('initialImageName')  # 初始图像
        target_images = request.form.get('targetImageName')  # 目标图像，即图像prompt

        user_id = session['user_id']
        if not checkIsLogin():
            # 判断用户是否登录
            # 用户需要先登录才能使用文本生成图像功能
            response_data = {
                'error': 'notLogin',
            }
            return jsonify(response_data)

        user_status, user_tcb = getUserInfo(user_id)
        # 判断用户的状态
        # 只有在先前提交的任务完成运行后，才可以继续提交下一个任务
        if user_status == UserStatus.HAS_WAIT_TASK:
            response_data = {
                'error': 'hasWaitTask',
            }
            return jsonify(response_data)
        elif user_status == UserStatus.HAS_FINISH_TASK:
            response_data = {
                'error': 'hasFinishTask',
            }
            return jsonify(response_data)
        elif user_status == UserStatus.HAS_RUNNING_TASK:
            response_data = {
                'error': 'hasRunningTask',
            }
            return jsonify(response_data)

        # 根据用户提交的表单生成对应的任务控制块
        tcb = TCB(user_id=user_id,
                  prompts=prompts, image_type=ImageType(image_type), iter_time=max_iterations,
                  aspect_ratio=AspectRatio(aspect_ratio), width=width, height=height,
                  generator=Generator(generator), discriminator=Discriminator(discriminator),
                  initial_image=initial_image, target_images=target_images)

        if painter is not None:
            # 若当前有其他用户的任务正在运行
            # 则将当前用户的任务加入等待队列
            waitingQueue.enqueue(tcb)
            # 同时向前端返回等待运行信息
            response_data = {
                'status': 'taskWait',
            }
            return jsonify(response_data)

        # painter==None
        # 当前没有其他用户的任务，可以直接执行当前用户提交的任务
        # 但是仍然需要先将tcb放入等待队列
        # 因为运行tcb由守护现场负责
        waitingQueue.enqueue(tcb)
        # 前端返回开始运行信息
        response_data = {
            'status': 'startGeneration',
        }
        return jsonify(response_data)


@blue_painter.route('/painter/generate-image', methods=['GET'])
def generate_image():
    # 返回模型运行时的相关状态
    # 如当前生成的图像、迭代次数、是否完成图像生成任务等

    global painter

    # 以下为线程控制块中的一些状态标识
    isStart = False  # 开始运行模型
    isInitial = False  # 模型正在初始化
    isImgGenerateStart = False  # 模型开始生成图像
    isVideoGenerateStart = False  # 模型开始生成视频
    isInterrupt = False  # 模型运行终止
    isFinish = False  # 模型完成运行
    curIter = 0

    # imagePath=="" 表示当前当前没有生成的图像
    # 由前端根据用户在表单中的选择显示默认图像
    imagePath = ""

    if painter is None:
        # 尚未运行模型或已经成功运行结束
        user_id = session['user_id']
        user_status, user_tcb = getUserInfo(user_id)
        if user_status == UserStatus.HAS_FINISH_TASK:
            # 若tcb成功运行结束，则设置isFinish标识为True
            # 同时根据tcb的状态设置isInterrupt标识
            # 若模型尚未运行，返回默认标识
            isInterrupt = user_tcb.taskStatus == TaskStatus.INTERRUPT
            isFinish = True
    else:
        # 模型开始运行
        isStart = True
        threadCTL = painter.threadCTL  # 获取模型的线程控制块

        curIter = painter.getCurIter()  # 获取模型的当前迭代次数
        if curIter == 0:
            # 模型处于初始化
            isInitial = True
        else:
            # 模型完成初始化，开始生成图像
            # 默认情况下，模型处于生成图像阶段
            isImgGenerateStart = True
            imagePath = f"/static/steps/{curIter:04d}.png"  # 根据模型的当前迭代次数，可以或许最新生成的图像

            if threadCTL['imageGenerated']:
                # 模型完成了 图像生成 或 图像生成终止
                isVideoGenerateStart = True

                if threadCTL['finish']:
                    # 图像完成 视频生成 或 图像生成终止
                    isFinish = True
                    if threadCTL['interrupt']:
                        isInterrupt = True
                        # 若模型正常运行结束，应当由守护线程将painter置为None，否则用户关闭网页后，painter永远不会被回收
                        # 若模型在生成图像的过程中终止，由于终止图像命令由用户主动发出，故由视图函数将painter置为None
                        painter = None

    # 将生成的图像通过base64编码后传给前端
    encoded_image = ""
    if imagePath != "":
        with open('.' + imagePath, "rb") as f:
            image_data = f.read()
        # 将图片数据转为base64编码
        encoded_image = base64.b64encode(image_data).decode('utf-8')

    # 传给前端的数据包含线程控制块和模型生成的最新图像
    response_data = {
        # "image_url": image_url,
        "image": encoded_image,
        'isStart': isStart,
        'isInitial': isInitial,
        'isImgGenerateStart': isImgGenerateStart,
        'isVideoGenerateStart': isVideoGenerateStart,
        'isInterrupt': isInterrupt,
        'isFinish': isFinish,
        'curIter': curIter,
    }
    return jsonify(response_data)


@blue_painter.route('/painter/interrupt-generate', methods=['POST', 'GET'])
def interrupt_generate():
    global painter

    user_id = session['user_id']
    user_status, user_tcb = getUserInfo(user_id)
    if painter is None:
        # 若painter==None，则代表等待队列为空，直接返回即可
        pass
    elif painter.tcb.user_id == user_id:
        # 中断当前正在运行的tcb
        painter.stop()
    else:
        # 从等待队列中将对应tcb移除
        waitingQueue.remove(user_id)
        user_tcb.taskStatus = TaskStatus.INTERRUPT

    response = make_response('', 204)
    return response


@blue_painter.route('/painter/saveImage', methods=['GET'])
def saveImage():
    """
    保存生成的图像
    """

    user_id = session['user_id']
    for tcb in finishedQueue:
        # 从完成队列中寻找用户的tcb
        if user_id == tcb.user_id:
            finishedQueue.remove(tcb)
            break
    else:
        # 没有在完成队列中找到用户tcb，向前端返回报错信息
        return jsonify({'success': False, 'error': 'notFound'})

    prompt = tcb.prompts

    prompts = [frase.strip() for frase in prompt.split("|")]
    prompt = prompts[0]
    tag = ''
    if len(prompts) > 1:
        tag = '|'.join(prompts[1:])

    # 向数据库中插入图片数据
    # 同时，图片在数据库中的id作为保存图片的名字
    with app.app_context():
        print(session['username'])
        user_id = User.query.filter(User.account == session['username']).first().id
        # image = Image(name=imgName, user_id=1)
        image = Image(name=prompt, iter_time=tcb.iter_time, image_type=tcb.image_type, tag=tag,
                      generator=tcb.generator, discriminator=tcb.discriminator,
                      aspect_ratio=tcb.aspect_ratio, width=tcb.width, height=tcb.height,
                      user_id=user_id)

        db.session.add(image)
        db.session.commit()
        destImgName = str(image.id)
        destVideoName = destImgName

    srcImagePath = os.path.join(tcb.tempResultDir, tcb.tempImageName)
    srcVideoPath = os.path.join(tcb.tempResultDir, tcb.tempVideoName)

    # 获取图片后缀 如.png
    imagePostfix = os.path.splitext(srcImagePath)[1]
    videoPostfix = os.path.splitext(srcVideoPath)[1]
    destImgName += imagePostfix
    destVideoName += videoPostfix

    # 要保存的图片位置
    destImgPath = os.path.join('.', imagesDirectory, destImgName)
    destVideoPath = os.path.join('.', videosDirectory, destVideoName)

    print('\n' + '-' * 10)
    print('Image Save...')
    print(f'Copy from {srcImagePath} to {destImgPath}')
    print('Video Save...')
    print(f'Copy from {srcVideoPath} to {destVideoPath}')
    print('-' * 10 + '\n')

    shutil.copy(srcImagePath, destImgPath)
    shutil.copy(srcVideoPath, destVideoPath)

    return jsonify({'success': True})


@blue_painter.route('/painter/clearImage', methods=['GET'])
def clearImage():
    """
    清除生成的图像
    """

    user_id = session['user_id']
    for tcb in finishedQueue:
        # 从完成队列中寻找用户tcb
        if user_id == tcb.user_id:
            finishedQueue.remove(tcb)
            break
    else:
        # 没有在完成队列中找到用户tcb，向前端返回报错信息
        return jsonify({'success': False, 'error': 'notFound'})

    srcImagePath = os.path.join(tcb.tempResultDir, tcb.tempImageName)

    print('\n' + '-' * 10)
    print('Image Clear...')
    print(f'Clear from {srcImagePath}')
    print('-' * 10 + '\n')

    os.remove(srcImagePath)

    return jsonify({'success': True})


@blue_painter.route('/painter/uploadInitialImage', methods=['POST'])
def uploadInitialImage():
    file = request.files['file']
    if file:
        print("用户上传初始文件：" + file.filename)
        # filename = secure_filename(file.filename)
        filename = file.filename

        # 获取文件后缀 如.png
        postfix = os.path.splitext(filename)[1]
        filepath = os.path.join('.', 'static/temp/' + str(session['user_id']), 'initialImage' + postfix)
        # 将初始图像保存退为 static/temp/user_id/initialImage.postfix
        file.save(filepath)

        response = make_response('', 200)
        return response
    else:
        response = make_response('', 204)
        return response


@blue_painter.route('/painter/uploadTargetImage', methods=['POST'])
def uploadTargetImage():
    file = request.files['file']
    if file:
        print("用户上传目标文件：" + file.filename)
        filename = file.filename

        # 获取文件后缀 如.png
        postfix = os.path.splitext(filename)[1]
        filepath = os.path.join('.', 'static/temp/' + str(session['user_id']), 'targetImage' + postfix)
        file.save(filepath)

        response = make_response('', 200)
        return response
    else:
        response = make_response('', 204)
        return response
