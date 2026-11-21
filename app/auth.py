"""The playground is open to guests; students log in with student number + name + password.

Anyone who is not logged in uses the playground as a guest (the front page, also after logging out):
no database, their runs train in the browser and live only in that page, and server training is for
logged-in users. Students come from the teacher's roster. The first login sets the password (typed
twice); after that the same password is required. A teacher can reset it.
There is no login page: a guest logs in from a popup on the playground (POST /login as JSON), and the
page then saves the runs made as a guest to that account (POST /api/runs) before it reloads.
"""
from functools import wraps

from flask import Blueprint, abort, g, jsonify, redirect, request, session, url_for
from flask_babel import gettext as _

from .extensions import db
from .i18n import LANGUAGES
from .models import User, utcnow

bp = Blueprint('auth', __name__)
MIN_PASSWORD_LENGTH = 4


@bp.before_app_request
def load_user():
    uid = session.get('uid')
    g.user = db.session.get(User, uid) if uid else None
    g.guest = g.user is None and session.get('guest') is True


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            if request.path.startswith('/api/'):
                if g.guest:            # a message, not 401 (the page would send the guest to the login form)
                    return jsonify({'errors': [_('This feature is not available to guests.')]}), 403
                abort(401)
            return redirect(login_url(request.path))
        return view(*args, **kwargs)
    return wrapped


def user_or_guest_required(view):
    """Pages and API calls a guest may use too (nothing that reads or writes the database).
    A page makes anyone not logged in a guest; an API call without a session (one that ran out) is a 401."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None and not g.guest:
            if request.path.startswith('/api/'):
                abort(401)
            start_guest()
        return view(*args, **kwargs)
    return wrapped


def login_url(next_path=None):
    """The playground with its login popup open; after the login the page goes to next_path."""
    return url_for('main.index', login=1, next=next_path or None)


def teacher_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if not g.user.is_teacher:
            abort(403)
        return view(*args, **kwargs)
    return wrapped


def normalize_name(name):
    return ''.join((name or '').split())


@bp.route('/login', methods=['GET', 'POST'])
def login():
    """POST as JSON from the login popup; anything else (an old bookmark or tab of the login page that
    was here before) goes to the playground with the popup open."""
    if request.method == 'POST' and request.is_json:
        return login_json()
    return redirect(login_url(request.args.get('next')))


def login_json():
    """The guest's login popup on the playground (POST /login as JSON, so the page and the guest's runs
    stay): {'user': id, 'home'} (the teacher page for a teacher, else null), or
    {'message', 'kind' ('error' | 'info'), 'first_login'} (to ask for the password again)."""
    body = request.get_json(silent=True)
    body = {k: v for k, v in body.items() if isinstance(v, str)} if isinstance(body, dict) else {}
    user, (category, message), first_login = check_login(
        body.get('login_id', '').strip(), body.get('name', '').strip(), body.get('password', ''),
        body.get('password_confirm'))
    if not user:
        return jsonify({'message': message, 'kind': category, 'first_login': first_login}), 400
    start_session(user)
    return jsonify({'user': user.id, 'home': url_for('admin.index') if user.is_teacher else None})


def check_login(login_id, name, password, confirm):
    """(user, (None, None), first_login) when they may log in (a first login has set the password),
    else (None, (flash category, message), first_login). confirm is None until the second password is asked."""
    user = find_user(login_id)
    if user is None or normalize_name(user.name) != normalize_name(name):
        return None, ('error', _('The student ID and name are not on the class list.')), False
    if user.has_password:
        if not user.check_password(password):
            return None, ('error', _('Wrong password. If you forgot it, ask your teacher to reset it.')), False
        return user, (None, None), False
    if confirm is None:
        return None, ('info', _('First login: type the same password again to register it.')), True
    if len(password) < MIN_PASSWORD_LENGTH:
        return None, ('error', _('Use a password of at least %(n)d characters.', n=MIN_PASSWORD_LENGTH)), True
    if password != confirm:
        return None, ('error', _('The two passwords are different.')), True
    user.set_password(password)
    return user, (None, None), True


def find_user(login_id):
    """By login ID; a school prefix is in small letters, so 'ABC21013' finds 'abc21013' too."""
    user = db.session.scalar(db.select(User).filter_by(login_id=login_id))
    if user is None and login_id != login_id.lower():
        user = db.session.scalar(db.select(User).filter_by(login_id=login_id.lower()))
    return user


def start_session(user):
    user.last_login_at = utcnow()
    db.session.commit()
    lang = session.get('lang')
    session.clear()
    session['uid'] = user.id
    if lang:
        session['lang'] = lang
    session.permanent = True


def start_guest():
    session['guest'] = True          # until the browser closes (not a permanent session); keeps the language
    g.guest = True


@bp.post('/logout')
def logout():
    lang = session.get('lang')
    session.clear()
    if lang:
        session['lang'] = lang
    start_guest()
    return redirect(url_for('main.index'))


@bp.get('/lang/<code>')
def set_language(code):
    if code in LANGUAGES:
        session['lang'] = code
    back = request.referrer or ''
    return redirect(back if back.startswith(request.host_url) else url_for('main.index'))
