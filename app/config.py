import os
from datetime import timedelta


class Config:
    APP_NAME = 'Machine Learning Playground'   # shown in every language
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///playground.db')   # file in instance/
    SESSION_COOKIE_SAMESITE = 'Lax'
    SESSION_COOKIE_HTTPONLY = True
    PERMANENT_SESSION_LIFETIME = timedelta(hours=12)
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024
    MAX_CONCURRENT_TRAININGS = int(os.environ.get('MAX_CONCURRENT_TRAININGS', 4))
    BABEL_DEFAULT_LOCALE = 'ko'
    BABEL_TRANSLATION_DIRECTORIES = 'translations'
