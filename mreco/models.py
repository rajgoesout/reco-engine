from datetime import datetime
from mreco import db, login
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from mongoengine.fields import SequenceField
from flask_mongoengine import BaseQuerySet


class Movie(db.DynamicDocument):
    meta = {
        'collection': 'movie',
        'queryset_class': BaseQuerySet
    }
    # movie_id = db.IntField(primary_key=True)


class User(UserMixin, db.Document):
    # id = db.IntField(primary_key=True)
    # user_id = db.IntField(autoincrement=True, primary_key=True, unique=True)
    # user_id = SequenceField(collection_name='user', db_alias='imovies', primary_key=True)
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
    # movie_id = db.ReferenceField(Movie)
    movie_id = db.IntField()
    user_id = db.ReferenceField(User)
    score = db.DecimalField()


# @login.user_loader
# def load_user(username):
#     # users = User.objects.all()
#     # if username not in users:
#     #     return
#     # user = User()
#     # user.id = username
#     # return user
#     # return User.objects(int(user_id))
#     return User.objects(username=username)

# @login.request_loader
# def request_loader(request):
#     email = request.form.get('email')
#     if email not in users:
#         return

#     user = User()
#     user.id = email

#     # DO NOT ever store passwords in plaintext and always compare password
#     # hashes using constant-time comparison!
#     user.is_authenticated = request.form['password'] == users[email]['password']

#     return user