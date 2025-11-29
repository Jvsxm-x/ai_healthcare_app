from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY','change-me')
DEBUG = True if os.environ.get('DJANGO_DEBUG','True')=='True' else False
MODEL_DIR = BASE_DIR
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin','django.contrib.auth','django.contrib.contenttypes','django.contrib.sessions',
    'django.contrib.messages','django.contrib.staticfiles',
    'rest_framework','corsheaders','api','channels',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware','django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware','django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware','django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware','django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'dawini.urls'
TEMPLATES = [{'BACKEND':'django.template.backends.django.DjangoTemplates','DIRS':[],'APP_DIRS':True,'OPTIONS':{'context_processors':['django.template.context_processors.debug','django.template.context_processors.request','django.contrib.auth.context_processors.auth','django.contrib.messages.context_processors.messages',],},},]
WSGI_APPLICATION = 'dawini.wsgi.application'
ASGI_APPLICATION = 'dawini.asgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# MongoDB Configuration - using pymongo directly
try:
    import pymongo
    # prefer MONGO_URI if provided, otherwise fall back to localhost
    MONGO_URI = 'mongodb://localhost:27017'
    if MONGO_URI:
        MONGO_CLIENT = pymongo.MongoClient(MONGO_URI)
        # if a default database is present in the URI, use it, otherwise use named db
        try:
            MONGO_DB = MONGO_CLIENT.get_default_database() or MONGO_CLIENT['dawini_db']
        except Exception:
            MONGO_DB = MONGO_CLIENT['dawini_db']
    else:
        MONGO_CLIENT = pymongo.MongoClient('localhost', 27017)
        MONGO_DB = MONGO_CLIENT['dawini_db']
except ImportError:
    # pymongo not installed — continue without MongoDB; set placeholders
    MONGO_CLIENT = None
    MONGO_DB = None
    MONGO_URI = os.environ.get('MONGO_URI', '')

REDIS_URL = os.environ.get('REDIS_URL','redis://localhost:6379')
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {"hosts": [REDIS_URL]},
    },
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = 'fr-fr'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True
STATIC_URL = '/static/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

CORS_ALLOW_ALL_ORIGINS = True

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticatedOrReadOnly',
    ),
}
from datetime import timedelta
SIMPLE_JWT = {'ACCESS_TOKEN_LIFETIME': timedelta(days=1),'AUTH_HEADER_TYPES': ('Bearer',),}
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')