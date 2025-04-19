import json

from flask import Blueprint, request, render_template, make_response, jsonify, session

blue_index = Blueprint("blue-index", __name__)

from app import db, app
from models import User

from constraint import cookieExpireTime
from utils import checkIsLogin


@blue_index.route('/')
@blue_index.route('/index')
def index():
    """
    index为网页首页
    """

    print(session.permanent)
    if checkIsLogin():
        username = session['username']
        response = make_response(render_template("index.html", username=username))
    else:
        response = make_response(render_template("index.html"))

    return response


@blue_index.route('/index/signin', methods=['POST'])
def signin():
    """
    用户登录
    """
    # from app import db, app
    content_type = request.headers.get('content-type')

    if content_type == 'application/json':
        data = json.loads(request.data.decode())
        account = data.get('account')
        password = data.get('password')

        print('\n' + '-' * 10)
        print("登录请求")
        print(account)
        print(password)
        print('-' * 10 + '\n')

        with app.app_context():
            user = User(account=account, password=password)
            query_result = User.query.filter(User.account == user.account).first()
            if query_result is None:
                # status=1 表示不存在该用户
                return jsonify({'success': False, 'status': 1}), 200
            elif query_result.password != user.password:
                # status=2 表示输入密码错误
                return jsonify({'success': False, 'status': 2}), 200

            # 将用户名和用户ID加入session，实现保持登录
            session['username'] = account
            session['user_id'] = query_result.id
            session.permanent = True

        # status=0 表示是合法的登录
        # 设置一些Cookie状态
        response = make_response(jsonify({'success': True, 'status': 0}), 200)
        response.set_cookie('isLogin', 'true', cookieExpireTime)
        response.set_cookie('hasSayHello', 'false', cookieExpireTime)
        response.set_cookie('username', account, cookieExpireTime)
        response.set_cookie('cookieExpireTime', str(cookieExpireTime), cookieExpireTime)
        return response
    else:
        return jsonify({'success': False, 'message': 'Unsupported Media Type'}), 415


@blue_index.route('/index/signup', methods=['POST'])
def signup():
    """
    用户注册
    """

    content_type = request.headers.get('content-type')

    if content_type == 'application/json':
        data = json.loads(request.data.decode())
        account = data.get('account')
        password = data.get('password')

        print('\n' + '-' * 10)
        print("注册请求")
        print(account)
        print(password)
        print('-' * 10 + '\n')

        with app.app_context():
            user = User(account=account, password=password)
            same_account_user = User.query.filter(User.account == user.account).all()
            if len(same_account_user) != 0:
                # status=1 表示该用户已存在
                return jsonify({'success': False, 'status': 1}), 200
            db.session.add(user)
            db.session.commit()

            # 将用户名和用户ID加入session
            session['username'] = account
            session['user_id'] = user.id
            session.permanent = True

        # status=0 表示是合法的登录
        response = make_response(jsonify({'success': True, 'status': 0}), 200)
        response.set_cookie('isLogin', 'true', cookieExpireTime)
        response.set_cookie('hasSayHello', 'false', cookieExpireTime)
        # response.set_cookie('username', account, cookieExpireTime)
        response.set_cookie('cookieExpireTime', str(cookieExpireTime), cookieExpireTime)
        return response
    else:
        return jsonify({'success': False, 'message': 'Unsupported Media Type'}), 415
