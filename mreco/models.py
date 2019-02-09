from mreco import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mongoengine import BaseQuerySet


class Movie(db.DynamicDocument):
    meta = {
        'collection': 'movie',
        'queryset_class': BaseQuerySet
    }


class User(UserMixin, db.Document):
    username = db.StringField(max_length=64, index=True, unique=True)
    email = db.StringField(max_length=120, index=True, unique=True)
    password_hash = db.StringField(max_length=128)

    def __repr__(self):
        return str(self.username)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Rating(db.Document):
    movie_id = db.IntField()
    user_id = db.ReferenceField(User)
    score = db.DecimalField()
