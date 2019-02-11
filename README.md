# mreco

[mreco](https://mreco.herokuapp.com/) is a Movie Recommendation System that leverages machine learning algorithms based on:

- User-user collaborative filtering
- Item-item collaborative filtering
- Dimensionality Reduction (Matrix Factorization)

### Getting started

Install a virtual environment on [mac/linux](https://www.codingforentrepreneurs.com/blog/install-django-on-mac-or-linux/) or [windows](https://www.codingforentrepreneurs.com/blog/install-python-django-on-windows)

```
$ pip install -r requirements.txt
```

Install [MongoDB](https://docs.mongodb.com/manual/installation/) and start a [mongod process](https://docs.mongodb.com/manual/tutorial/manage-mongodb-processes/). Use the collection dump provided in the repository.

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

- Flask (python web framework)
- MongoDB (document store)
- mongoengine (Document-Object Mapper)
- scikit-learn, pandas, numpy, scipy (for machine learning and data cleaning)
- mlab (Database-as-a-Service for MongoDB, used in production)
- Heroku (Cloud Application Platform)
- Bootstrap (Frontend Framework)

### Directory Structure

```bash
.
├── Procfile
├── README.md
├── data.json
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
│       ├── movie.html
│       └── user.html
├── r.csv
├── r2.csv
├── requirements.txt
├── runtime.txt
└── u.csv
```

### Credits

Inspired by [https://towardsdatascience.com/the-4-recommendation-engines-that-can-predict-your-movie-tastes-109dc4e10c52](https://towardsdatascience.com/the-4-recommendation-engines-that-can-predict-your-movie-tastes-109dc4e10c52) and [https://towardsdatascience.com/how-to-build-a-simple-song-recommender-296fcbc8c85](https://towardsdatascience.com/how-to-build-a-simple-song-recommender-296fcbc8c85)
