from flask import Flask
from flask import render_template
from flask_mongoengine import MongoEngine
from flask_login import (
    LoginManager, UserMixin, login_user, login_required, logout_user, current_user
)

db = MongoEngine()


class Movie(db.DynamicDocument):
    meta = {
        'collection': 'imcoll'
    }
    rating = db.IntField()


class User(UserMixin, db.Document):
    meta = {'collection': 'usercoll'}
    username = db.StringField(max_length=30)
    password = db.StringField()
