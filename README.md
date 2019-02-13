# mreco

[mreco](https://mreco.herokuapp.com/) is a Movie Recommendation System that leverages algorithms based on:

- User-user collaborative filtering (CF)
- Item-item collaborative filtering (CF)
- Dimensionality Reduction (Matrix Factorization)

### Getting started

Install a virtual environment on [mac/linux](https://www.codingforentrepreneurs.com/blog/install-django-on-mac-or-linux/) or [windows](https://www.codingforentrepreneurs.com/blog/install-python-django-on-windows)

```
$ pip install -r requirements.txt
```

Install [MongoDB](https://docs.mongodb.com/manual/installation/) and start a [mongod process](https://docs.mongodb.com/manual/tutorial/manage-mongodb-processes/). Use the collection dump provided in the repository. [Instructions](https://docs.mongodb.com/manual/tutorial/backup-and-restore-tools/)

You can also run this command to import movies from `data.json` file:

```
$ mongoimport --db imovies --collection movie --file data.json --jsonArray
```

Set environment variables:

```
$ export FLASK_ENV=development
$ export FLASK_APP=mreco
```

Run the app:

```
$ flask run
```

### Dependencies

- Web Framework - [Flask](http://flask.pocoo.org/)
- Database - [MongoDB](https://www.mongodb.com/)
- Document-Object Mapper - [mongoengine](http://mongoengine.org/)
- ML and data science libraries - scikit-learn, pandas, numpy, scipy
- Database Service (used in production) - [mlab](https://mlab.com/)
- Cloud Application Platform - [Heroku](https://www.heroku.com/)
- Frontend Framework - [Bootstrap](https://getbootstrap.com/)

### Directory Structure

```
.
├── Dockerfile
├── Procfile
├── README.md
├── data.json
├── docker-compose.yml
├── dump
│   └── imovies
│       ├── movie.bson
│       ├── movie.metadata.json
│       ├── rating.bson
│       ├── rating.metadata.json
│       ├── user.bson
│       └── user.metadata.json
├── instance
├── m.csv
├── mreco
│   ├── __init__.py
│   ├── forms.py
│   ├── models.py
│   ├── recommender.py
│   ├── routes.py
│   ├── static
│   │   ├── logo.png
│   │   └── mr.png
│   └── templates
│       ├── auth
│       │   ├── login.html
│       │   └── register.html
│       ├── base.html
│       ├── index.html
│       └── movie.html
├── r.csv
├── r2.csv
├── requirements.txt
├── runtime.txt
└── u.csv
```

### Approach followed

- Scraped movies data from Imdb and saved it in a json file.
- Imported the json file into a mongodb collection.
- Build a flask webapp with user authentication, CRUD (rating movies).
- `mreco/recommender.py` contains 3 algorithms:

  - Popularity based recommender:
    It recommends the most popular movies, same recommendations to all the users. I have just written the algorithm but not used it in the actual app.
  - Item similarity recommender (CF):
    It computes similarity between items and recommends items that are similar to an item that is liked by a particular user. Here, `item` means a movie. Suppose `u` is the currently logged in user.
    High level pseudocode of the algorithm:

    ```
    for every item i that u has no preference for yet:
      for every item j that u has a preference for:
        compute a similarity s between i and j
        add u's preference for j, weighted by s, to a running average
    return the top items, ranked by weighted average
    ```

  - User similarity recommender (CF):
    It computes similarity between users and recommends items liked by a given user to the other users who are similar to the given user. Suppose `u` is the currently logged in user.
    High level pseudocode of the algorithm:

    ```
    for every item i that u has no preference for yet:
      for every other user v that has a preference for i:
        compute a similarity s between u and v
        add v's preference for i, weighted by s, to a running average
    return the top items, ranked by weighted average
    ```

- `mreco/routes.py` contains all the url routes, and also a method `matrix_factorization` where the Dimensionality Reduction (low ranked matrix factorization) based algorithm has been implemented. I have used Singular Value Decomposition (SVD) to create a low ranked matrix.

### Tasks

- [x] Scrape Imdb and build mongodb collection
- [x] Create web app
- [x] Implement Collaborative filtering (CF) algorithms
- [x] Implement Matrix Factorization algorithm
- [x] Deploy on heroku

### Credits

Inspired by https://towardsdatascience.com/the-4-recommendation-engines-that-can-predict-your-movie-tastes-109dc4e10c52 and https://towardsdatascience.com/how-to-build-a-simple-song-recommender-296fcbc8c85
