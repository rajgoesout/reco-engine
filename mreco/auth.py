import functools

from flask import (
    Flask, Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import Email, Length, InputRequired
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import (
    LoginManager, UserMixin, login_user, login_required, logout_user, current_user
)
from flask_mongoengine import MongoEngine

from .db import User


bp = Blueprint('auth', __name__, url_prefix='/auth')

# login_manager = LoginManager()
# login_manager.init_app(app)
# login_manager.login_view = 'login'

db = MongoEngine()


# @login_manager.user_loader
# def load_user(user_id):
#     return User.objects(pk=user_id).first()


class RegForm(FlaskForm):
    username = StringField('username',  validators=[
        InputRequired(),  Length(max=30)])
    password = PasswordField('password', validators=[
                             InputRequired(), Length(min=8, max=20)])


@bp.route('/register', methods=('GET', 'POST'))
def register():
    form = RegForm()
    if request.method == 'POST':
        if form.validate():
            existing_user = User.objects(email=form.email.data).first()
            if existing_user is None:
                hashpass = generate_password_hash(
                    form.password.data, method='sha256')
                hey = User(form.username.data, hashpass).save()
                login_user(hey)
                return redirect(url_for('auth.login'))
            else:
                return 'user exists already'
        else:
            return 'invalid form'
    return render_template('auth/register.html', form=form)


@bp.route('/login', methods=('GET', 'POST'))
def login():
    if current_user.is_authenticated == True:
        return redirect(url_for('index'))
        # return redirect('/')
    form = RegForm()
    if request.method == 'POST':
        if form.validate():
            check_user = User.objects(username=form.username.data).first()
            if check_user:
                if check_password_hash(check_user['password'], form.password.data):
                    login_user(check_user)
                    return redirect(url_for('index'))
                    # return redirect('/')
                else:
                    print('correct')
            else:
                print('not check_user')
        else:
            return 'invalid form'
    return render_template('auth/login.html', form=form)


@bp.route('/logout')
@login_required
def logout():
    session.clear()
    return redirect(url_for('index'))
