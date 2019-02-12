import os

from flask import Flask
from flask_mongoengine import MongoEngine
from flask_login import LoginManager


test_config = None

app = Flask(__name__, instance_relative_config=True)
app.config.from_object(__name__)

if os.environ.get('FLASK_ENV') == 'development':
    app.config['TESTING'] = True
    app.debug = True
    app.config['MONGODB_SETTINGS'] = {
        'db': 'imovies',
    }
else:
    app.config['MONGODB_DB'] = 'imovies'
    app.config['MONGODB_HOST'] = 'mongodb://<your_mlab_username>:<your_mlab_password>@ds227255.mlab.com:27255/imovies'
    app.config['MONGODB_PORT'] = 27255
    app.config['MONGODB_USERNAME'] = '<your_mlab_username>'
    app.config['MONGODB_PASSWORD'] = '<your_mlab_password>'
    app.config['TESTING'] = False
    app.debug = False

app.config['SECRET_KEY'] = 'thisismysecret'

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


login = LoginManager(app)
login.login_view = 'login'


from mreco import routes, models


if __name__ == '__main__':
    app.run()
