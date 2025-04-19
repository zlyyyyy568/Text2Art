from flask import session


def checkIsLogin():
    return 'username' in session