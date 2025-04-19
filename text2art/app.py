import os
from datetime import timedelta

from flask import Flask, render_template, jsonify, request, make_response, send_file, \
    session
from flask_sqlalchemy import SQLAlchemy

from PIL import Image
from utils import checkIsLogin

app = Flask(__name__)

app.config['SECRET_KEY'] = "maple"
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)

# 设置session永久
# 这个设置好像需要通过插件才能实现，在Flask中没有这个配置
# app.config['SESSION_PERMANENT'] = True

# 让 session cookie 在用户关闭浏览器时自动过期
# app.config['SESSION_COOKIE_EXPIRES'] = timedelta(seconds=0)

# 2.设置数据库的配置信息
# 设置数据库的链接信息,
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:123456@127.0.0.1:3306/text2art"
# 该字段增加了大量的开销,会被禁用,建议设置为False
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# 3.创建sqlalchemy对象db,关联app
db = SQLAlchemy(app)

with app.app_context():
    # 注册蓝图
    from index import blue_index
    from painter import blue_painter
    from homepage import blue_homepage, filter_square
    from gallery import blue_gallery

    app.register_blueprint(blue_index)
    app.register_blueprint(blue_painter)
    app.register_blueprint(blue_homepage)
    app.register_blueprint(blue_gallery)


@app.route('/about')
def about():
    return 'test'


@app.route('/test', methods=['POST', 'GET'])
def test():
    return render_template('test.html')


# @app.route('/download')
# def video():
#     filePath = request.args.get('filePath')
#     startIndex = filePath.find('static')
#     filePath = filePath[startIndex:]
#
#     return send_file(filePath, as_attachment=True)


@app.route('/logout')
def logout():
    """
    用户登出，返回首页
    """

    # 从Session中将用户移除
    session.pop('username', None)
    session.pop('user_id', None)

    # 用户退出登录后回到首页
    response = make_response(render_template("index.html"))
    # 删除用户的所有Cookie
    for cookie_name in request.cookies:
        response.delete_cookie(cookie_name)
    return response


@app.route('/uploadProfile', methods=['POST'])
def uploadProfile():
    """
    处理用户上传头像请求
    """

    file = request.files['file']  # 获得用户上传的文件
    if file:
        print("用户上传头像：" + file.filename)
        filename = file.filename

        # 获取文件后缀 如.png
        postfix = os.path.splitext(filename)[1]

        # 判断用户上传的文件是否合法
        if postfix not in ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.gif', '.pcx', '.tga', '.exif', '.fpx', '.svg',
                           '.psd', '.cdr', '.pcd', '.dxf', '.ufo', '.eps', '.ai', '.raw', '.WMF', '.webp', '.avif',
                           '.apng']:
            return make_response('', 206)

        if not os.path.exists('static/temp/' + str(session['user_id'])):
            os.mkdir('static/temp/' + str(session['user_id']))
        filepath = os.path.join('.', 'static/temp/' + str(session['user_id']), 'profileImage' + postfix)
        file.save(filepath)  # 将用户上传的头像保存至服务器

        # 为便于在前端中显示，将用户上传的头像裁剪为正方形
        img = Image.open(filepath)
        w, h = img.size
        if w < h:
            offset = (h - w) // 2
            box = (0, offset, w, h - offset)
            img_crop = img.crop(box)
        else:
            offset = (w - h) // 2
            box = (offset, 0, w - offset, h)
            img_crop = img.crop(box)

        os.remove(filepath)
        filepath = os.path.join('.', 'static/temp/' + str(session['user_id']), 'profileImage.png')
        img_crop.save(filepath, 'PNG')

        response = make_response('', 200)
        return response
    else:
        response = make_response('', 204)
        return response


app.add_template_filter(filter_square, 'filter_square')



if __name__ == '__main__':
    print(app.url_map)
    with app.app_context():
        db.create_all()
    print(app.permanent_session_lifetime)

    app.run(debug=True, host='0.0.0.0')