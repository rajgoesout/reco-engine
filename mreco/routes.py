import math

from flask import render_template, flash, redirect, url_for, request, session
from flask_login import login_user, logout_user, current_user
from werkzeug.urls import url_parse
from mongoengine import DoesNotExist

from mreco import app, login
from mreco.forms import LoginForm, RegistrationForm, RatingForm
from mreco.models import Movie, User, Rating
from mreco.recommender import (
    item_similarity_recommender_py, user_similarity_recommender_py)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse.linalg import svds


def generate_csv():
    import csv
    with open('r.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['uid', 'movie_id', 'score'])
        for r in Rating.objects.all():
            writer.writerow([int(str(r.user_id.id), 16), r.movie_id, r.score])
    rdf = pd.read_csv('r.csv', usecols=['uid', 'movie_id', 'score'])

    with open('u.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['uid'])
        for u in User.objects.all():
            writer.writerow([int(str(u.id), 16)])
    udf = pd.read_csv('u.csv', usecols=['uid'])
    udf.insert(0, 'user_id', range(1, 1+len(udf)))
    rdf = pd.merge(udf, rdf[['uid', 'movie_id', 'score']], on='uid')
    print(rdf)
    rdf.to_csv('r2.csv', index=False)
    return rdf


def predict(ratings, similarity, typex='user'):
    """Function to predict ratings"""
    if typex == 'user':
        # print(dir(ratings))
        mean_user_rating = ratings.mean(axis=1)
        # Use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(
            ratings_diff)/np.array([np.abs(similarity).sum(axis=1)]).T
    elif typex == 'item':
        pred = ratings.dot(
            similarity)/np.array([np.abs(similarity).sum(axis=1)])
    return pred


def rmse(pred, actual):
    """Function to calculate RMSE"""
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return math.sqrt(mean_squared_error(pred, actual))


def similarity():
    """user-user and item-item collaborative filtering algorithm."""
    movies = pd.read_csv(
        'm.csv',
        usecols=['movie_id', 'title', 'genres'],
        dtype={'movie_id': 'int32', 'title': 'str', 'genres': 'str'})
    ratings = generate_csv()
    print(ratings)
    train_data, test_data = train_test_split(ratings, test_size=0.2)
    print('train_data:')
    print(train_data)
    print('test_data:')
    print(test_data)
    train_data_matrix = train_data.as_matrix(
        columns=['user_id', 'movie_id', 'score'])
    test_data_matrix = test_data.as_matrix(
        columns=['user_id', 'movie_id', 'score'])
    print(train_data_matrix.shape)
    print(test_data_matrix.shape)
    user_correlation = 1 - pairwise_distances(train_data, metric='correlation')
    user_correlation[np.isnan(user_correlation)] = 0
    print(user_correlation[:4, :4])
    item_correlation = 1 - \
        pairwise_distances(train_data_matrix.T, metric='correlation')
    item_correlation[np.isnan(item_correlation)] = 0
    print(item_correlation[:4, :4])

    # Predict ratings on the training data with both similarity score
    user_prediction = predict(
        train_data_matrix, user_correlation, typex='user')
    item_prediction = predict(
        train_data_matrix, item_correlation, typex='item')
    print(user_prediction)
    print(item_prediction)
    print(pd.DataFrame(user_prediction))
    print(pd.DataFrame(item_prediction))

    # RMSE on the test data
    print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
    print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

    # RMSE on the train data
    print('User-based CF RMSE: ' + str(rmse(user_prediction, train_data_matrix)))
    print('Item-based CF RMSE: ' + str(rmse(item_prediction, train_data_matrix)))


def user_user():
    ratings = generate_csv()
    n_movies = ratings.movie_id.unique().shape[0]
    print('n_movs:'+str(n_movies))
    this_u = ratings.loc[ratings['uid'] == str(
        int(session['user_id'], 16)), 'user_id'].iloc[0]
    print(this_u)
    k = ratings[ratings.user_id == this_u].shape[0]
    print(k)

    aggregation_functions = {'score': 'sum'}
    ratings_grouped = ratings.groupby(ratings['movie_id']).aggregate(
        aggregation_functions).reset_index()
    grouped_sum = ratings_grouped['score'].sum()
    ratings_grouped['percentage'] = ratings_grouped['score'].div(
        grouped_sum)*100
    ratings_grouped.sort_values(['score', 'movie_id'],
                                ascending=[0, 1]).reset_index()
    print(ratings_grouped)

    users = ratings['user_id'].unique()
    movies = ratings['movie_id'].unique()

    train_data, test_data = train_test_split(
        ratings, test_size=0.20, random_state=0)

    is_model = user_similarity_recommender_py()
    is_model.create(ratings, 'user_id', 'movie_id')

    predictions = is_model.recommend(this_u, k=k)
    print(predictions)

    return predictions


def item_item():
    ratings = generate_csv()
    n_movies = ratings.movie_id.unique().shape[0]
    print('n_movs:'+str(n_movies))
    this_u = ratings.loc[ratings['uid'] == str(
        int(session['user_id'], 16)), 'user_id'].iloc[0]
    print(this_u)
    k = ratings[ratings.user_id == this_u].shape[0]
    print(k)

    aggregation_functions = {'score': 'sum'}
    ratings_grouped = ratings.groupby(ratings['movie_id']).aggregate(
        aggregation_functions).reset_index()
    grouped_sum = ratings_grouped['score'].sum()
    ratings_grouped['percentage'] = ratings_grouped['score'].div(
        grouped_sum)*100
    ratings_grouped.sort_values(['score', 'movie_id'],
                                ascending=[0, 1]).reset_index()
    print(ratings_grouped)

    users = ratings['user_id'].unique()
    movies = ratings['movie_id'].unique()

    train_data, test_data = train_test_split(
        ratings, test_size=0.20, random_state=0)

    is_model = item_similarity_recommender_py()
    is_model.create(ratings, 'user_id', 'movie_id')

    predictions = is_model.recommend(this_u, k=k)
    print(predictions)

    return predictions


def recommend_movies(predictions, userID, movies, original_ratings, num_recommendations):
    """Get and sort the user's predictions"""
    user_row_number = userID - 1  # User ID starts at 1, not 0
    sorted_user_predictions = predictions.iloc[user_row_number].sort_values(
        ascending=False
    )  # User ID starts at 1

    # Get the user's data and merge in the movie information.
    user_data = original_ratings[original_ratings.user_id == (userID)]
    user_full = (
        user_data.merge(
            movies, how='left', left_on='movie_id', right_on='movie_id'
        ).sort_values(['score'], ascending=False)
    )

    print('User {0} has already rated {1} movies.'.format(
        userID, user_full.shape[0]))
    num_recommendations = int(user_full.shape[0])
    print('Recommending highest {0} predicted ratings movies not already rated.'.format(
        num_recommendations))

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (
        movies[~movies['movie_id'].isin(user_full['movie_id'])].merge(
            pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
            left_on='movie_id',
            right_on='movie_id'
        ).rename(
            columns={user_row_number: 'Predictions'}
        ).sort_values(
            'Predictions', ascending=False
        ).iloc[:num_recommendations, :-1]
    )

    return user_full, recommendations


def matrix_factorization():
    movies = pd.read_csv(
        'm.csv',
        usecols=['movie_id', 'title', 'genres'],
        dtype={'movie_id': 'int32', 'title': 'str', 'genres': 'str'})
    ratings = generate_csv()
    print(movies)
    print(ratings)
    n_users = ratings.user_id.unique().shape[0]
    n_movies = ratings.movie_id.unique().shape[0]
    print('Number of users = ' + str(n_users) +
          ' | Number of movies = ' + str(n_movies))
    Ratings = ratings.pivot(
        index='user_id', columns='movie_id', values='score').fillna(0)
    print(Ratings)
    R = Ratings.as_matrix()
    user_ratings_mean = np.mean(R, axis=1)
    Ratings_demeaned = R - user_ratings_mean.reshape(-1, 1)
    sparsity = round(1.0 - len(ratings) / float(n_users * n_movies), 3)
    print('The sparsity level of the dataset is ' +
          str(sparsity * 100) + '%')
    U, sigma, Vt = svds(Ratings_demeaned, k=1)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(
        np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds = pd.DataFrame(all_user_predicted_ratings, columns=Ratings.columns)
    print(preds)
    userID = ratings.loc[ratings['uid'] ==
                         str(int(str(session['user_id']), 16)), 'user_id'].iloc[0]
    already_rated, predictions = recommend_movies(
        preds, userID, movies, ratings, n_movies)
    print(already_rated)
    print(predictions)

    return [already_rated, predictions]


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
        Rating.objects.count()
        try:
            uu_list = user_user()
            print(uu_list)
            uu_rec = list(uu_list.to_dict()['movie'].values())
            print(uu_rec)
            uu_movies = []
            for uuu in uu_rec:
                uu_movies.append(Movie.objects.get(movie_id=uuu))
            for i in range(len(uu_movies)):
                print(uu_movies[i].title)
        except (pd.core.base.DataError, IndexError, ValueError, NameError, KeyError) as e:
            print(e)
            uu_rec = []
            uu_movies = []
        print('uu_count='+str(len(uu_movies)))
        try:
            ii_list = item_item()
            print(ii_list)
            ii_rec = list(ii_list.to_dict()['movie'].values())
            print(ii_rec)
            ii_movies = []
            for iiu in ii_rec:
                ii_movies.append(Movie.objects.get(movie_id=iiu))
            for i in range(len(ii_movies)):
                print(ii_movies[i].title)
        except(pd.core.base.DataError, IndexError, ValueError, NameError, KeyError) as e:
            print(e)
            ii_rec = []
            ii_movies = []
        print('ii_count='+str(len(ii_movies)))
        try:
            mf_list = matrix_factorization()
            print(mf_list[1])
            mf_rec = list(mf_list[1].to_dict()['movie_id'].values())
            print(mf_rec)
            mf_movies = []
            for mfu in mf_rec:
                mf_movies.append(Movie.objects.get(movie_id=mfu))
            for i in range(len(mf_movies)):
                print(mf_movies[i].title)
        except (IndexError, ZeroDivisionError, ValueError, NameError, KeyError):
            mf_rec = []
            mf_movies = []
        print(len(mf_movies))
        return render_template(
            'index.html',
            uu_count=len(uu_movies), uu_movies=uu_movies, uu_rec=uu_rec,
            ii_count=len(ii_movies), ii_movies=ii_movies, ii_rec=ii_rec,
            mf_count=len(mf_movies), mf_movies=mf_movies, mf_rec=mf_rec,
            title='Home', this_u=this_u, current_user=current_user, movies=movies)
    except KeyError:
        return redirect(url_for('login'))


@app.route('/movies/<int:movie_id>', methods=['GET', 'POST'])
def rate_movie(movie_id):
    user_id = session['user_id']
    print(user_id)
    movie = Movie.objects.get(movie_id=movie_id)
    this_u = User.objects(id=session['user_id'])
    form = RatingForm()
    if form.validate_on_submit():
        try:
            rating = Rating.objects.get(
                user_id=user_id, movie_id=movie_id)
            rating.score = form.score.data
        except DoesNotExist:
            rating = Rating.objects.create(
                score=form.score.data,
                user_id=user_id,
                movie_id=movie_id)
        print(movie['id'])
        print(movie_id)
        print(rating.save())
        flash('Your rating has been submitted successfully!')
        print(rating.score, rating.user_id.id, rating.movie_id)
    return render_template('movie.html', this_u=this_u, form=form, current_user=current_user, movie=movie)


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
        login_user(user, remember=form.remember_me.data, force=True)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
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
