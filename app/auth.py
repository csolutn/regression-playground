"""Login with student number + name + password.

Students come from the teacher's roster. The first login sets the password
(typed twice); after that the same password is required. A teacher can reset it.
"""
from functools import wraps

from flask import Blueprint, abort, flash, g, make_response, redirect, render_template, request, session, url_for
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


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            if request.path.startswith('/api/'):
                abort(401)
            return redirect(url_for('auth.login', next=request.path))
        return view(*args, **kwargs)
    return wrapped


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
    form = {'login_id': '', 'name': ''}
    first_login, kept_password = False, ''
    if request.method == 'POST':
        form = {'login_id': request.form.get('login_id', '').strip(), 'name': request.form.get('name', '').strip()}
        password = request.form.get('password', '')
        user = db.session.scalar(db.select(User).filter_by(login_id=form['login_id']))

        if user is None or normalize_name(user.name) != normalize_name(form['name']):
            flash(_('The student number and name do not match the class list. Ask your teacher.'), 'error')
        elif not user.has_password:
            first_login = True
            confirm = request.form.get('password_confirm')
            if confirm is None:
                kept_password = password    # refilled so only the confirmation needs typing
                flash(_('First login: type the same password again to register it.'), 'info')
            elif len(password) < MIN_PASSWORD_LENGTH:
                flash(_('Use a password of at least %(n)d characters.', n=MIN_PASSWORD_LENGTH), 'error')
            elif password != confirm:
                flash(_('The two passwords are different.'), 'error')
            else:
                user.set_password(password)
                return finish_login(user)
        elif not user.check_password(password):
            flash(_('Wrong password. If you forgot it, ask your teacher to reset it.'), 'error')
        else:
            return finish_login(user)
    resp = make_response(render_template('login.html', form=form, first_login=first_login,
                                         kept_password=kept_password))
    resp.headers['Cache-Control'] = 'no-store'     # the page may carry the typed password
    return resp


def finish_login(user):
    user.last_login_at = utcnow()
    db.session.commit()
    next_url, lang = request.args.get('next', ''), session.get('lang')
    session.clear()
    session['uid'] = user.id
    if lang:
        session['lang'] = lang
    session.permanent = True
    if next_url.startswith('/') and not next_url.startswith('//'):
        return redirect(next_url)
    return redirect(url_for('admin.index' if user.is_teacher else 'main.index'))


@bp.post('/logout')
def logout():
    lang = session.get('lang')
    session.clear()
    if lang:
        session['lang'] = lang
    return redirect(url_for('auth.login'))


@bp.get('/lang/<code>')
def set_language(code):
    if code in LANGUAGES:
        session['lang'] = code
    back = request.referrer or ''
    return redirect(back if back.startswith(request.host_url) else url_for('main.index'))
