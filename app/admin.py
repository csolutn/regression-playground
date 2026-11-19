"""Teacher pages: class list, password reset, and every student's history.
The first teacher (the oldest teacher account) also adds and deletes the other teachers; a new teacher, like a
student, sets the password at the first login."""
import csv
import io
from datetime import timezone
from zoneinfo import ZoneInfo

from functools import wraps

from flask import Blueprint, Response, abort, flash, g, redirect, render_template, request, url_for
from flask_babel import gettext as _

from .auth import teacher_required
from .extensions import db
from .ml import options
from .models import Run, User
from .roster import import_roster, parse_roster

bp = Blueprint('admin', __name__, url_prefix='/teacher')


def run_counts():
    return dict(db.session.execute(db.select(Run.user_id, db.func.count(Run.id)).group_by(Run.user_id)).all())


def all_students():
    return db.session.scalars(db.select(User).filter_by(role='student').order_by(User.login_id)).all()


def all_teachers():
    """Oldest first: the first one is the first teacher."""
    return db.session.scalars(db.select(User).filter_by(role='teacher').order_by(User.id)).all()


def first_teacher_required(view):
    @wraps(view)
    @teacher_required
    def wrapped(*args, **kwargs):
        if all_teachers()[0] != g.user:
            abort(403)
        return view(*args, **kwargs)
    return wrapped


def csv_response(rows, filename):
    """UTF-8 with BOM so Excel shows Korean correctly."""
    out = io.StringIO()
    out.write('\ufeff')
    csv.writer(out).writerows(rows)
    return Response(out.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename={filename}'})


@bp.get('/')
@teacher_required
def index():
    last = dict(db.session.execute(db.select(Run.user_id, db.func.max(Run.created_at)).group_by(Run.user_id)).all())
    teachers = all_teachers()
    return render_template('admin/index.html', students=all_students(), counts=run_counts(), last=last,
                           teachers=teachers if teachers[0] == g.user else None)


@bp.get('/students.csv')
@teacher_required
def export_students():
    """The class list with each student's number of runs, as on the teacher page."""
    counts = run_counts()
    return csv_response([[_('Student ID'), _('Name'), _('Runs')],
                         *([s.login_id, s.name, counts.get(s.id, 0)] for s in all_students())], 'students.csv')


@bp.post('/roster')
@teacher_required
def upload_roster():
    file = request.files.get('roster')
    if not file or not file.filename:
        flash(_('Choose a CSV file.'), 'error')
        return redirect(url_for('.index'))
    raw = file.read()
    for encoding in ('utf-8-sig', 'cp949'):     # Excel in Korea often saves CSV as CP949
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        flash(_('The file could not be read. Save it as CSV (UTF-8).'), 'error')
        return redirect(url_for('.index'))
    added, updated = import_roster(parse_roster(text))
    flash(_('Class list imported: %(added)d added, %(updated)d updated.', added=added, updated=updated), 'info')
    return redirect(url_for('.index'))


@bp.post('/students')
@teacher_required
def add_student():
    login_id, name = request.form.get('login_id', '').strip(), request.form.get('name', '').strip()
    if not login_id or not name:
        flash(_('Enter a student number and a name.'), 'error')
    else:
        added, updated = import_roster([(login_id, name)])
        flash(_('Student added.') if added else _('Student updated.'), 'info')
    return redirect(url_for('.index'))


def get_student(user_id):
    user = db.get_or_404(User, user_id)
    if user.is_teacher:
        abort(404)
    return user


@bp.post('/students/<int:user_id>/reset-password')
@teacher_required
def reset_password(user_id):
    user = get_student(user_id)
    user.password_hash = None
    db.session.commit()
    flash(_('Password reset for %(name)s. The next login sets a new one.', name=user.name), 'info')
    return redirect(url_for('.index'))


@bp.post('/students/<int:user_id>/delete')
@teacher_required
def delete_student(user_id):
    user = get_student(user_id)
    db.session.delete(user)
    db.session.commit()
    flash(_('Deleted %(name)s and their history.', name=user.name), 'info')
    return redirect(url_for('.index'))


@bp.post('/teachers')
@first_teacher_required
def add_teacher():
    login_id, name = request.form.get('login_id', '').strip(), request.form.get('name', '').strip()
    if not login_id or not name:
        flash(_('Enter a login ID and a name.'), 'error')
    elif db.session.scalar(db.select(User.id).filter_by(login_id=login_id)) is not None:
        flash(_('The login ID %(login_id)s is already in use.', login_id=login_id), 'error')
    else:
        db.session.add(User(login_id=login_id, name=name, role='teacher'))      # no password: set at the first login
        db.session.commit()
        flash(_('Teacher %(name)s added. The first login sets the password.', name=name), 'info')
    return redirect(url_for('.index'))


def get_other_teacher(user_id):
    user = db.get_or_404(User, user_id)
    if not user.is_teacher or user == g.user:          # the first teacher keeps its account and password
        abort(404)
    return user


@bp.post('/teachers/<int:user_id>/reset-password')
@first_teacher_required
def reset_teacher_password(user_id):
    user = get_other_teacher(user_id)
    user.password_hash = None
    db.session.commit()
    flash(_('Password reset for the teacher %(name)s. The next login sets a new one.', name=user.name), 'info')
    return redirect(url_for('.index'))


@bp.post('/teachers/<int:user_id>/delete')
@first_teacher_required
def delete_teacher(user_id):
    user = get_other_teacher(user_id)
    db.session.delete(user)
    db.session.commit()
    flash(_('Deleted %(name)s and their history.', name=user.name), 'info')
    return redirect(url_for('.index'))


@bp.get('/students/<int:user_id>')
@teacher_required
def student(user_id):
    return render_template('admin/student.html', student=get_student(user_id), options=options.public_options())


@bp.get('/runs.csv')
@teacher_required
def export_runs():
    """All runs as a spreadsheet. Times are in ?tz= (the teacher's browser time zone, added by the page), else UTC."""
    try:
        zone = ZoneInfo(request.args.get('tz') or 'UTC')
    except (KeyError, ValueError):                # unknown or malformed zone name
        zone = ZoneInfo('UTC')
    keys = ['function_1d', 'function_2d', 'data_size', 'noise_std', 'model', 'hidden_layers', 'activation',
            'max_depth', 'n_estimators', 'loss', 'optimizer', 'learning_rate', 'epochs', 'batch_method']
    rows = [['student_id', 'name', 'run', f'created_at ({zone.key})', 'status', 'final_train_loss', 'final_val_loss',
             'duration_s', 'n_inputs', 'data_source'] + keys]
    for run, user in db.session.execute(db.select(Run, User).join(User).order_by(User.login_id, Run.seq)):
        cfg = run.config
        created = run.created_at.replace(tzinfo=timezone.utc).astimezone(zone)   # stored as naive UTC
        rows.append([user.login_id, user.name, run.seq, f'{created:%Y-%m-%d %H:%M:%S}', run.status, run.final_train,
                     run.final_val, run.duration, cfg.get('n_inputs'), cfg.get('data_source')] + [cfg.get(k) for k in keys])
    return csv_response(rows, 'runs.csv')
