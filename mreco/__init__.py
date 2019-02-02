import os

from flask import Flask
from flask import render_template
from flask_mongoengine import MongoEngine
from flask_login import (
    LoginManager, UserMixin, login_user, login_required, logout_user, current_user
)

from .db import Movie
from . import auth


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(__name__)
    app.config['MONGODB_SETTINGS'] = {
        'db': 'imovies',
    }
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'wtfiswrongwithya'
    app.debug = True

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    db = MongoEngine()
    db.init_app(app)

    app.register_blueprint(auth.bp)

    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'

    @login_manager.user_loader
    def load_user(user_id):
        return User.objects(pk=user_id).first()

    @app.route('/')
    def index():
        movies = Movie.objects.all()
        return render_template('index.html', movies=movies, user=current_user)

    return app
