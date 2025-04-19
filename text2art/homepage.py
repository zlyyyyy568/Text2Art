import json

from flask import Blueprint, request, render_template, make_response, jsonify, session

blue_homepage = Blueprint("blue-homepage", __name__)

from app import db, app
from models import User, Image, Comment

from utils import checkIsLogin


class JinJaImage:
    def __init__(self, image: Image):
        self.id = image.id
        self.name = image.name
        self.image_type = image.image_type.value
        self.generator = image.generator.value
        self.discriminator = image.discriminator.value

        self.aspect_ratio = image.aspect_ratio.value
        self.width = image.width
        self.height = image.height
        self.iter_time = image.iter_time

        self.tag = image.tag

        self.user_id = image.user_id
        self.user_name = "该账户已注销"
        self.imageSrc = f'static/results/images/{self.id}.png'

        self.time = image.time.strftime('%Y年%m月%d日')
        self.like = image.like


def filter_square(li):
    temp_li = []

    for image in li:
        if image.height == 512:
            temp_li.append(image)
    return temp_li


@blue_homepage.route('/homepage')
def index():
    user_id = session['user_id']
    username = session['username']

    with app.app_context():
        images = Image.query.filter(Image.user_id == user_id).all()
        jinjaimages = []
        for image in images:
            jinjaimages.append(JinJaImage(image))

    return render_template('homepage.html', username=username, user_id=user_id, images=jinjaimages)


@blue_homepage.route('/homepage/deleteImage', methods=['POST'])
def deleteImage():
    content_type = request.headers.get('content-type')

    if content_type == 'application/json':
        user_id = session['user_id']
        username = session['username']

        data = json.loads(request.data.decode())
        imageID = data.get('imageID')

        print('\n' + '-' * 10)
        print("删除图片")
        print(imageID)
        print('-' * 10 + '\n')

        with app.app_context():
            image = Image.query.filter(Image.id == imageID).first()
            comments = Comment.query.filter(Comment.image_id == imageID).all()

            for comment in comments:
                db.session.delete(comment)

            if image is None:
                response = make_response(jsonify({'success': False, 'status': 0}), 200)
                return response
            else:
                db.session.delete(image)
                db.session.commit()

        # status=0 表示是合法的登录
        response = make_response(jsonify({'success': True, 'status': 0}), 200)

        return response
    else:
        return jsonify({'success': False, 'message': 'Unsupported Media Type'}), 415


@blue_homepage.route('/homepage/showProfile')
def showProfile():
    with app.app_context():
        user_id = session['user_id']
        images = Image.query.filter(Image.user_id == user_id).order_by(Image.id.desc()).all()
        imagesSrc = [f"/static/results/images/{image.id}.png" for image in images]
        comments = Comment.query.filter(Comment.user_id == user_id).all()

    data = {
        'createNum': len(images),
        'commentNum': len(comments),
        'images': imagesSrc[0:4],
    }

    return jsonify(data)
