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
import numpy as np
import os
import math
# from flask_security.forms import Form, LoginForm
from surprise import Reader, Dataset, SVD, evaluate, model_selection
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse.linalg import svds


def generate_csv():
    import csv
    with open('r.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['uid', 'movie_id', 'score'])
        for r in Rating.objects.all():
            writer.writerow([r.user_id.id, r.movie_id, r.score])
    rdf=pd.read_csv('r.csv',usecols=['uid','movie_id','score'])

    with open('u.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['uid'])
        for u in User.objects.all():
            writer.writerow([u.id])
    udf=pd.read_csv('u.csv',usecols=['uid'])
    udf.insert(0,'user_id',range(1,1+len(udf)))
    rdf=pd.merge(udf,rdf[['uid','movie_id','score']],on='uid')
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
        pred = ratings.dot(similarity)/np.array([np.abs(similarity).sum(axis=1)])
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
    generate_csv()
    ratings = pd.read_csv(
        'r.csv',
        usecols=['user_id', 'movie_id', 'score'],
        # dtype={'user_id': 'int32', 'movie_id': 'int32', 'score': 'float'}
    )
    train_data, test_data = train_test_split(ratings, test_size=0.5)
    print(train_data)
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
    user_prediction = predict(train_data_matrix, user_correlation, typex='user')
    item_prediction = predict(train_data_matrix, item_correlation, typex='item')
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


def recommend_movies(predictions, userID, movies, original_ratings, num_recommendations):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # User ID starts at 1, not 0
    # print(dir(predictions))
    # print('itrows: '+str(predictions.iterrows()))
    # print(original_ratings.loc[original_ratings['user_id']==userID])
    sorted_user_predictions = predictions.iloc[user_row_number].sort_values(ascending=False) # User ID starts at 1
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings[original_ratings.user_id == (userID)]
    user_full = (user_data.merge(movies, how = 'left', left_on = 'movie_id', right_on = 'movie_id').
                     sort_values(['score'], ascending=False)
                 )

    print('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print('Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations))
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies[~movies['movie_id'].isin(user_full['movie_id'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movie_id',
               right_on = 'movie_id').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations


def matrix_factorization():
    movies = pd.read_csv(
        'm.csv',
        usecols=['movie_id', 'title', 'genres'],
        dtype={'movie_id': 'int32', 'title': 'str', 'genres': 'str'})
    # generate_csv()
    # ratings = pd.read_csv(
    #     'r.csv',
    #     usecols=['user_id', 'movie_id', 'score'],
    #     # dtype={'user_id': 'int32', 'movie_id': 'int32', 'score': 'float'}
    # )
    ratings = generate_csv()
    print(movies)
    print(ratings)
    # ratings.to_csv('r2.csv', index=False)
    n_users = ratings.user_id.unique().shape[0]
    n_movies = ratings.movie_id.unique().shape[0]
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_movies))
    Ratings = ratings.pivot(index='user_id', columns='movie_id', values='score').fillna(0)
    print(Ratings)
    R = Ratings.as_matrix()
    user_ratings_mean = np.mean(R, axis = 1)
    Ratings_demeaned = R - user_ratings_mean.reshape(-1, 1)
    sparsity = round(1.0 - len(ratings) / float(n_users * n_movies), 3)
    print('The sparsity level of MovieLens1M dataset is ' +  str(sparsity * 100) + '%')
    U, sigma, Vt = svds(Ratings_demeaned, k = 1)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds = pd.DataFrame(all_user_predicted_ratings, columns = Ratings.columns)
    print(preds)
    print('this is wuw')
    userID = ratings.loc[ratings['uid']==session['user_id'], 'user_id'].iloc[0]
    print(type(int(userID)))
    already_rated, predictions = recommend_movies(preds, userID, movies, ratings, n_movies)
    print(already_rated)
    print(predictions)

    return [already_rated, predictions]


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
    data.split(n_folds=df_rating.count())
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
        try:
            # mylist = real_stuff()
            print(mylist[5])
            rec4u = list(mylist[3].to_dict()['movie_id'].values())
            print(list(mylist[3].to_dict()['movie_id'].values()))
            ur_movies = []
            for ru in rec4u:
                ur_movies.append(Movie.objects.get(movie_id=ru))
            for i in range(len(ur_movies)):
                print(ur_movies[i].title)
        except pd.core.base.DataError and ValueError and NameError:
            rec4u = []
            ur_movies = []
        # similarity()
        mf_list = matrix_factorization()
        print(mf_list[1])
        mf_rec = list(mf_list[1].to_dict()['movie_id'].values())
        print(mf_rec)
        mf_movies = []
        for mfu in mf_rec:
            mf_movies.append(Movie.objects.get(movie_id=mfu))
        for i in range(len(mf_movies)):
            print(mf_movies[i].title)
        # train_data, test_data = train_test_split(
        #     df_rating, test_size=0.20, random_state=0)
        # pm = popularity_recommender_py()
        # pm.create(train_data, 'user_id', 'movie_id')
        # is_model = item_similarity_recommender_py()
        # is_model.create(train_data, 'user_id', 'movie_id')
        # print(is_model.recommend(session['user_id']))
        # print(pm.recommend(this_u))
        return render_template('index.html', mf_movies=mf_movies, mf_rec=mf_rec, ur_movies=ur_movies, rec4u=rec4u, title='Home', this_u=this_u, current_user=current_user, movies=movies)
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
            # rating = Rating.objects(
            #     (Rating.user_id == user_id) and (Rating.movie_id == movie_id)).first()
            # if rating.user_id == user_id:
            rating.score = form.score.data
            #     print("LG")
            # else:
            #     print('else')
            #     rating = Rating.objects.create(score=form.score.data,
            #                                    user_id=user_id, movie_id=movie_id)
        except DoesNotExist:
            # del rating
            print('dne')
            rating = Rating.objects.create(score=form.score.data,
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
        print(rating.save())
        # print(dir(rating))
        print(rating.score, rating.user_id.id, rating.movie_id)
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
