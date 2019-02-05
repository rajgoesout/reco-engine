import os

# from . import routes, models
from flask import Flask, render_template
from flask_mongoengine import MongoEngine, MongoEngineSessionInterface
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_mongo_sessions import MongoDBSessionInterface


test_config = None

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

db = MongoEngine()
db.init_app(app)
# app.session_interface = MongoEngineSessionInterface(db)
# app.session_interface = MongoDBSessionInterface(app, db, 'sessions')


login = LoginManager(app)
login.login_view = 'login'


from mreco import routes, models
# from mreco.models import Movie


# @app.route('/')
# @app.route('/index')
# def index():
#     movies = Movie.objects.all()
#     return render_template('index.html', title='Home', movies=movies)

if __name__ == '__main__':
    app.run()
