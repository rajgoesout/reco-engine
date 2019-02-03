from flask import render_template, flash, redirect, url_for, request, session
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.urls import url_parse
from mreco import app, db, login
from mreco.forms import LoginForm, RegistrationForm
from mreco.models import Movie, User, Rating


@login.user_loader
def load_user(username):
    # users = User.objects.all()
    # if username not in users:
    #     return
    # user = User()
    # user.id = username
    # return user
    # return User.objects(int(user_id))
    return User.objects(username=username)


@app.route('/')
@app.route('/index')
def index():
    movies = Movie.objects.all()
    if session:
        print(session['user_id'])
        this_u = User.objects(id=session['user_id'])
        return render_template('index.html', title='Home', this_u=this_u, current_user=current_user, movies=movies)
    else:
        return render_template('index.html', title='Home', this_u=current_user, current_user=current_user, movies=movies)


@app.route('/users/<username>', methods=['GET'])
def show_user(username):
    users = User.objects.all()
    user = User.objects.get(username=username)
    return render_template('user.html', user=user)


@app.route('/movies/<int:movie_id>', methods=['GET'])
def show_movie(movie_id):
    movies = Movie.objects.all()
    movie = Movie.objects.get(movie_id=movie_id)
    return render_template('movie.html', current_user=current_user, movie=movie)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
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


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'GET':
#         return '''
#                <form action='login' method='POST'>
#                 <input type='text' name='email' id='email' placeholder='email'/>
#                 <input type='password' name='password' id='password' placeholder='password'/>
#                 <input type='submit' name='submit'/>
#                </form>
#                '''

#     email = request.form['email']
#     users = User.objects.all()
#     if request.form['password'] == users[email]['password']:
#         user = User()
#         user.id = email
#         login_user(user)
#         return redirect(flask.url_for('protected'))

#     return 'Bad login'

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
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
