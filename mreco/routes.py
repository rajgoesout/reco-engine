from flask import render_template, flash, redirect, url_for, request, session
from flask_login import login_user, logout_user, current_user
from werkzeug.urls import url_parse
from mreco import app, db, login
from mreco.forms import LoginForm, RegistrationForm, RatingForm
from mreco.models import Movie, User, Rating
from mreco.recommender import popularity_recommender_py, item_similarity_recommender_py
from flask_mongoengine.wtf import model_form
from mongoengine import DoesNotExist
from flask_security import (
    Security, MongoEngineUserDatastore, UserMixin, RoleMixin, login_required)
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import pandas as pd
import os
# from flask_security.forms import Form, LoginForm
from surprise import Reader, Dataset, SVD, evaluate, model_selection


def generate_csv():
    import csv
    with open('r.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['user_id', 'movie_id', 'score'])
        for r in Rating.objects.all():
            writer.writerow([r.user_id.id, r.movie_id, r.score])


def real_stuff():
    # print('printing')
    df_movies = pd.read_csv(
        # os.path.join(data_path, movies_filename),
        'm.csv',
        usecols=['movie_id', 'title', 'genres'],
        dtype={'movie_id': 'int32', 'title': 'str', 'genres': 'str'})
    # print('printing')
    print(df_movies.head())
    generate_csv()
    df_rating = pd.read_csv(
        'r.csv',
        usecols=['user_id', 'movie_id', 'score'],
        # dtype={'user_id'}
    )
    print(df_rating.head())
    movie_data = pd.merge(df_rating, df_movies, on='movie_id')
    print(movie_data.head())
    print(movie_data.groupby('title')[
          'score'].mean().sort_values(ascending=False).head())
    print(movie_data.groupby('title')[
          'score'].count().sort_values(ascending=False).head())
    ratings_mean_count = pd.DataFrame(
        movie_data.groupby('title')['score'].mean())
    ratings_mean_count['rating_counts'] = pd.DataFrame(
        movie_data.groupby('title')['score'].count())
    print(ratings_mean_count)
    this_user_movie_data = movie_data.loc[movie_data['user_id']
                                          == session['user_id']]
    print(this_user_movie_data)
    # pivot ratings into movie features
    # df_movie_features = df_rating.pivot(
    #     index='movie_id',
    #     columns='user_id',
    #     values='score'
    # ).fillna(0)
    # convert dataframe of movie features to scipy sparse matrix
    # mat_movie_features = csr_matrix(df_movie_features.values)
    # print(df_movie_features.head())
    # print(mat_movie_features)

    # model_knn = NearestNeighbors(
    #     metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    # neigh = KNeighborsClassifier(n_neighbors=20)

    # train_data, test_data = train_test_split(
    #     df_rating, test_size=0.20, random_state=0)
    # pm = popularity_recommender_py()
    # pm.create(train_data, 'user_id', 'movie_id')
    # pm.recommend()
    reader = Reader()
    data = Dataset.load_from_df(
        df_rating[['user_id', 'movie_id', 'score']], reader)
    data.split(n_folds=5)
    svd = SVD()
    print(model_selection.cross_validate(svd, data, measures=['RMSE', 'MAE']))
    trainset = data.build_full_trainset()
    svd.fit(trainset)
    print(df_rating[df_rating['user_id'] == '5c5708f4bb9e3176d7d04cd4'])
    print(svd.predict('5c5708f4bb9e3176d7d04cd4', 4))
    return [df_movies, df_rating, movie_data, this_user_movie_data,
            movie_data.groupby('title')['score'].mean(
            ).sort_values(ascending=False),
            this_user_movie_data.groupby('title')['score'].mean().sort_values(ascending=False)]


@login.user_loader
def load_user(username):
    return User.objects(username=username)


@app.route('/')
@app.route('/index')
def index():
    movies = Movie.objects.all()
    try:
        a = session['user_id']
        print(a)
        print(session['user_id'])
        this_u = User.objects(id=session['user_id'])
        k = Rating.objects.count()
        mylist = real_stuff()
        print(mylist[5])
        rec4u = list(mylist[3].to_dict()['movie_id'].values())
        print(list(mylist[3].to_dict()['movie_id'].values()))
        ur_movies = []
        for ru in rec4u:
            ur_movies.append(Movie.objects.get(movie_id=ru))
        for i in range(len(ur_movies)):
            print(ur_movies[i].title)
        # train_data, test_data = train_test_split(
        #     df_rating, test_size=0.20, random_state=0)
        # pm = popularity_recommender_py()
        # pm.create(train_data, 'user_id', 'movie_id')
        # is_model = item_similarity_recommender_py()
        # is_model.create(train_data, 'user_id', 'movie_id')
        # print(is_model.recommend(session['user_id']))
        # print(pm.recommend(this_u))
        return render_template('index.html', ur_movies=ur_movies, rec4u=rec4u, title='Home', this_u=this_u, current_user=current_user, movies=movies)
    except KeyError:
        # return 'no'
        return redirect(url_for('login'))
    # if session:
    #     # if 'user_id' not in session.keys():
    #     #     redirect(url_for('login'))
    #     # if g.user is None:
    #         # raise RequestRedirect(url_for('login'))
    #     print(session['user_id'])
    #     this_u = User.objects(id=session['user_id'])
    #     k = Rating.objects.count()
    #     mylist = real_stuff()
    #     print(mylist[5])
    #     rec4u = list(mylist[3].to_dict()['movie_id'].values())
    #     print(list(mylist[3].to_dict()['movie_id'].values()))
    #     ur_movies = []
    #     for ru in rec4u:
    #         ur_movies.append(Movie.objects.get(movie_id=ru))
    #     for i in range(3):
    #         print(ur_movies[i].title)
    #     # train_data, test_data = train_test_split(
    #     #     df_rating, test_size=0.20, random_state=0)
    #     # pm = popularity_recommender_py()
    #     # pm.create(train_data, 'user_id', 'movie_id')
    #     # is_model = item_similarity_recommender_py()
    #     # is_model.create(train_data, 'user_id', 'movie_id')
    #     # print(is_model.recommend(session['user_id']))
    #     # print(pm.recommend(this_u))
    #     return render_template('index.html', ur_movies=ur_movies, rec4u=rec4u, title='Home', this_u=this_u, current_user=current_user, movies=movies)
    # else:
    #     print('anonymous')
    #     return render_template('index.html', title='Home', this_u=current_user, current_user=current_user, movies=movies)


@app.route('/users/<username>', methods=['GET'])
def show_user(username):
    users = User.objects.all()
    user = User.objects.get(username=username)
    return render_template('user.html', user=user)


@app.route('/movies/<int:movie_id>', methods=['GET', 'POST'])
def rate_movie(movie_id):
    user_id = session['user_id']
    print(user_id)
    movie = Movie.objects.get(movie_id=movie_id)
    form = RatingForm()
    # new_score = request.form.get("movie_rating")
    # new_score = form.score.data
    # if request.method == 'POST' and form.validate():
    if form.validate_on_submit():
        # rating = None
        # rating = Rating.objects(
        #     (Rating.user_id == user_id) and (Rating.movie_id == movie_id)).first()
        # rating = Rating.objects(movie_id=movie_id).first()
        # print(rating)
        try:
            rating = Rating.objects.get(
                user_id=user_id, movie_id=movie_id)
            rating.score = form.score.data
            print("LG")
        except DoesNotExist:
            # del rating
            rating = Rating(score=form.score.data,
                            user_id=user_id, movie_id=movie_id)
        print(movie['id'])
        print(movie_id)
        # if rating:
        # rating.score = form.score.data
        # else:
        # rating = Rating(score=form.score.data,
        #                 user_id=user_id, movie_id=movie_id)
        # else:
        # rating = Rating.objects.get(user_id=user_id, movie_id=movie['id'])
        # rating.score = form.score.data
        rating.save()
        print(rating.score)
    else:
        print('inv')
    return render_template('movie.html', form=form, current_user=current_user, movie=movie)
    # new_score = request.form.get("movie-rating")
    # rating = Rating.objects(
    #     (Rating.user_id == user_id) and (Rating.movie_id == movie_id)).first()

    # if not rating:
    #     rating = Rating(score=new_score, user_id=user_id, movie_id=movie_id)
    # else:
    #     rating.score = new_score
    # rating.save()
    # movies = Movie.objects.all()
    # movie = Movie.objects.get(movie_id=movie_id)
    # return redirect('/movies/%s' % movie_id)
    # return render_template('movie.html', current_user=current_user, movie=movie)


@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        if current_user.is_authenticated:
            return redirect(url_for('index'))
    except AttributeError:
        pass
    form = LoginForm()
    if form.validate_on_submit():
        user = User.objects.filter(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        v = login_user(user, remember=form.remember_me.data, force=True)
        print(v)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    else:
        print('invalid')
    return render_template('auth/login.html', title='Sign In', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    try:
        if current_user.is_authenticated:
            return redirect(url_for('index'))
    except AttributeError:
        pass
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        user.save()
        # db.post_save(user)
        # db.session.add(user)
        # db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('auth/register.html', title='Register', form=form)
