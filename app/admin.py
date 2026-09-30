"""Teacher pages: class list, password reset, and every student's history."""
import csv
import io

from flask import Blueprint, Response, abort, flash, redirect, render_template, request, url_for
from flask_babel import gettext as _

from .auth import teacher_required
from .extensions import db
from .ml import options
from .models import Run, User
from .roster import import_roster, parse_roster

bp = Blueprint('admin', __name__, url_prefix='/teacher')


@bp.get('/')
@teacher_required
def index():
    counts = dict(db.session.execute(db.select(Run.user_id, db.func.count(Run.id)).group_by(Run.user_id)).all())
    last = dict(db.session.execute(db.select(Run.user_id, db.func.max(Run.created_at)).group_by(Run.user_id)).all())
    students = db.session.scalars(db.select(User).filter_by(role='student').order_by(User.login_id)).all()
    return render_template('admin/index.html', students=students, counts=counts, last=last)


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


@bp.get('/students/<int:user_id>')
@teacher_required
def student(user_id):
    return render_template('admin/student.html', student=get_student(user_id), options=options.public_options())


@bp.get('/runs.csv')
@teacher_required
def export_runs():
    """All runs as a spreadsheet (UTF-8 with BOM so Excel shows Korean correctly)."""
    out = io.StringIO()
    out.write('﻿')
    w = csv.writer(out)
    keys = ['function_1d', 'function_2d', 'data_size', 'noise_std', 'model', 'hidden_layers', 'activation',
            'max_depth', 'n_estimators', 'loss', 'optimizer', 'learning_rate', 'epochs', 'batch_method']
    w.writerow(['student_id', 'name', 'run', 'created_at', 'status', 'final_train_loss', 'final_val_loss',
                'duration_s', 'n_inputs', 'data_source'] + keys)
    rows = db.session.execute(db.select(Run, User).join(User).order_by(User.login_id, Run.seq))
    for run, user in rows:
        cfg = run.config
        w.writerow([user.login_id, user.name, run.seq, run.created_at.isoformat(), run.status, run.final_train,
                    run.final_val, run.duration, cfg.get('n_inputs'), cfg.get('data_source')] + [cfg.get(k) for k in keys])
    return Response(out.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=runs.csv'})
