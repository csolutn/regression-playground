"""Language selection and the translation catalog for JavaScript.

Source strings are English. Korean lives in app/translations/ko/LC_MESSAGES/messages.po.
JavaScript calls t('English text'); the browser receives the whole catalog as JSON.
"""
from flask import session
from flask_babel import get_translations

LANGUAGES = {'ko': '한국어', 'en': 'English'}
DEFAULT_LANGUAGE = 'ko'


def select_locale():
    lang = session.get('lang')
    return lang if lang in LANGUAGES else DEFAULT_LANGUAGE


def js_catalog():
    catalog = getattr(get_translations(), '_catalog', {}) or {}
    return {k: v for k, v in catalog.items() if isinstance(k, str) and k and v}
