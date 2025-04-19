import json
import os.path

from flask import Blueprint, request, render_template, make_response, jsonify, session

blue_gallery = Blueprint("blue-gallery", __name__)

from app import db, app
from models import User, Image, Comment
from homepage import JinJaImage

from utils import checkIsLogin


@blue_gallery.route('/gallery')
def gallery():
    if checkIsLogin():
        user_id = session.get('user_id')
        username = session.get('username')
        # with app.app_context():
        #     images = Image.query.order_by(Image.id.asc()).paginate(page=1, per_page=9)
        #     print('\n' + '-' * 10)
        #     print(images.items)
        #     print('-' * 10)
        #     # images = Image.query.all()
        #     jinjaimages = []
        #     for image in images:
        #         jinjaimage = JinJaImage(image)
        #         user = User.query.filter(User.id == jinjaimage.user_id).first()
        #         if user is not None:
        #             jinjaimage.user_name = user.account
        #         jinjaimages.append(jinjaimage)
        #
        # return render_template('gallery.html', username=username, user_id=user_id, images=jinjaimages)
        return render_template('gallery.html', username=username, user_id=user_id)

    return render_template('gallery.html')


@blue_gallery.route('/gallery/get_more_images')
def get_more_images():
    page = int(request.args.get('page'))
    per_page = int(request.args.get('perPage'))
    # 注意，isTop是字符串类型
    isTop = request.args.get('isTop')
    with app.app_context():
        if isTop == 'true':
            images = Image.query.order_by(Image.like.desc()).order_by(Image.id.asc()).paginate(page=page,
                                                                                               per_page=per_page)
        else:
            images = Image.query.order_by(Image.time.desc()).order_by(Image.id.desc()).paginate(page=page,
                                                                                                per_page=per_page)

        jinjaimages = []
        for image in images.items:
            # jinjaimages.append(JinJaImage(image))
            jinjaimage = JinJaImage(image)
            user = User.query.filter(User.id == jinjaimage.user_id).first()
            if user is not None:
                jinjaimage.user_name = user.account
            jinjaimages.append(jinjaimage)

    isFullyLoad = (len(images.items) != per_page)

    return jsonify({
        'images': [image.__dict__ for image in jinjaimages],
        'isFullyLoad': isFullyLoad
    })


@blue_gallery.route('/gallery/sendComment', methods=['POST'])
def sendComment():
    data = json.loads(request.data.decode())
    with app.app_context():
        comment = Comment(text=data.get('message'), image_id=data.get('image_id'), user_id=session['user_id'])
        db.session.add(comment)
        db.session.commit()

    return jsonify({
        'message': '插入评论'
    })


class CommentData:
    pass


@blue_gallery.route('/gallery/getComments')
def getComments():
    image_id = int(request.args.get('imageID'))

    with app.app_context():
        comments = Comment.query.filter(image_id == Comment.image_id).order_by(Comment.time.desc()).all()
        commentDatas = []
        for i in range(len(comments)):
            commentData = CommentData()
            commentData.id = comments[i].id
            commentData.imgsrc = '/static/resource/profile-photo.avif'
            if os.path.exists(f'static/temp/{comments[i].user_id}/profileImage.png'):
                commentData.imgsrc = f'static/temp/{comments[i].user_id}/profileImage.png'
            commentData.userName = User.query.filter(User.id == comments[i].user_id).first().account
            commentData.text = comments[i].text
            commentData.time = comments[i].time.strftime('%Y年 %m月%d日 %H:%M')

            commentDatas.append(commentData)

    return jsonify({
        'comments': [comment.__dict__ for comment in commentDatas],
    })


@blue_gallery.route('/gallery/deleteComment')
def deleteComment():
    commentID = int(request.args.get('commentID'))

    with app.app_context():
        db.session.delete(Comment.query.filter(commentID == Comment.id).first())
        db.session.commit()

    return make_response('', 204)


@blue_gallery.route('/gallery/addLike')
def addLike():
    image_id = int(request.args.get('imageID'))

    with app.app_context():
        image = Image.query.filter(Image.id == image_id).first()
        image.like += 1
        db.session.commit()
        newLikeNum = image.like

    return jsonify({
        'newLikeNum': newLikeNum,
    })
