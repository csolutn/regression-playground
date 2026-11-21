"""Teacher pages: class list, password reset, and the students' history (results only; a backup as one HTML file).
The system admin (the oldest teacher account) also adds, edits, resets and deletes the other teachers; a new
teacher, like a student, sets the password at the first login. A teacher from another school gets a prefix
(access.py: who sees which students)."""
import csv
import io
import re
from functools import wraps

from flask import Blueprint, Response, abort, flash, g, jsonify, redirect, render_template, request, url_for
from flask_babel import gettext as _

from . import report
from .access import can_see, is_system_admin, students_of
from .auth import teacher_required
from .bundle import bundle
from .extensions import db
from .ml import options
from .models import Run, User
from .roster import import_roster, parse_roster

bp = Blueprint('admin', __name__, url_prefix='/teacher')
PREFIX = re.compile(r'[a-z]{2,10}')
# what a teacher types to delete all their students (in either language; the dialog shows the page's one)
DELETE_WORDS = ('삭제', 'delete')
BACKUP_WORDS = ('백업 후 삭제', 'back up and delete')        # the page saves the backup first
# the HTML backup's script: report-offline.js and what it imports, each after its imports (bundle.py)
REPORT_MODULES = ['i18n', 'describe', 'plot', 'history', 'report', 'nn', 'replay-here', 'report-offline']


def run_counts():
    return dict(db.session.execute(db.select(Run.user_id, db.func.count(Run.id)).group_by(Run.user_id)).all())


def my_students():
    return db.session.scalars(students_of(g.user)).all()


def all_teachers():
    """Oldest first: the first one is the system admin."""
    return db.session.scalars(db.select(User).filter_by(role='teacher').order_by(User.id)).all()


def system_admin_required(view):
    @wraps(view)
    @teacher_required
    def wrapped(*args, **kwargs):
        if not is_system_admin(g.user):
            abort(403)
        return view(*args, **kwargs)
    return wrapped


def csv_response(rows, filename):
    """UTF-8 with BOM so Excel shows Korean correctly."""
    out = io.StringIO()
    out.write('﻿')
    csv.writer(out).writerows(rows)
    return Response(out.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename={filename}'})


def only_theirs(teacher):
    """A prefixed teacher's students that no other teacher has: deleted with the teacher."""
    return [s for s in teacher.students if s.teachers == [teacher]] if teacher.prefix else []


@bp.get('/')
@teacher_required
def index():
    last = dict(db.session.execute(db.select(Run.user_id, db.func.max(Run.created_at)).group_by(Run.user_id)).all())
    teachers = all_teachers() if is_system_admin(g.user) else None
    return render_template('admin/index.html', students=my_students(), counts=run_counts(), last=last,
                           teachers=teachers, only_theirs={t.id: len(only_theirs(t)) for t in teachers or []},
                           student_counts={t.id: len(t.students) for t in teachers or []},
                           own_class=len(own_class(g.user)), delete_words=DELETE_WORDS, backup_words=BACKUP_WORDS,
                           options=options.public_options())


@bp.get('/students.csv')
@teacher_required
def export_students():
    """The class list with each student's number of runs, as on the teacher page (student numbers without a prefix)."""
    counts = run_counts()
    return csv_response([[_('Student ID'), _('Name'), _('Runs')],
                         *([s.number, s.name, counts.get(s.id, 0)] for s in my_students())], 'students.csv')


@bp.get('/roster-example.csv')
@teacher_required
def roster_example():
    """A class list to fill in: English column names (they survive any encoding) and one example row."""
    return csv_response([['student_id', 'name'], ['21013', '홍길동']], 'class_list_example.csv')


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
    added, updated = import_roster(parse_roster(text), g.user)
    flash(_('Class list imported: %(added)d added, %(updated)d updated.', added=added, updated=updated), 'info')
    return redirect(url_for('.index'))


@bp.post('/students')
@teacher_required
def add_student():
    login_id, name = request.form.get('login_id', '').strip(), request.form.get('name', '').strip()
    if not login_id or not name:
        flash(_('Enter a student number and a name.'), 'error')
    else:
        added, updated = import_roster([(login_id, name)], g.user)
        flash(_('Student added.') if added else _('Student updated.'), 'info')
    return redirect(url_for('.index'))


def get_student(user_id):
    user = db.get_or_404(User, user_id)
    if user.is_teacher or not can_see(g.user, user):
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
    if g.user in user.teachers and len(user.teachers) > 1:      # another teacher has them too: off my list only
        user.teachers.remove(g.user)
        db.session.commit()
        flash(_('Removed %(name)s from your list (another teacher still has them).', name=user.name), 'info')
        return redirect(url_for('.index'))
    db.session.delete(user)
    db.session.commit()
    flash(_('Deleted %(name)s and their history.', name=user.name), 'info')
    return redirect(url_for('.index'))


def own_class(teacher):
    """What "delete all students" deletes: a prefixed teacher's list, else the home school's students (the
    system admin removes another school with its teacher)."""
    if teacher.prefix:
        return list(teacher.students)
    return db.session.scalars(db.select(User).filter_by(role='student', prefix='')).all()


@bp.post('/students/delete-all')
@teacher_required
def delete_all_students():
    typed = ' '.join(request.form.get('confirm_text', '').split()).lower()
    if typed not in DELETE_WORDS + BACKUP_WORDS:
        flash(_('Nothing was deleted: the phrase did not match.'), 'error')
        return redirect(url_for('.index'))
    deleted = removed = 0
    for user in own_class(g.user):
        if g.user in user.teachers and len(user.teachers) > 1:   # another teacher has them too: off my list only
            user.teachers.remove(g.user)
            removed += 1
        else:
            db.session.delete(user)
            deleted += 1
    db.session.commit()
    flash(_('Deleted %(n)d students and their history.', n=deleted) if not removed else
          _('Deleted %(n)d students and their history; %(m)d others stay on another teacher\'s list.', n=deleted, m=removed), 'info')
    return redirect(url_for('.index'))


@bp.get('/students/<int:user_id>')
@teacher_required
def student(user_id):
    return render_template('admin/student.html', student=get_student(user_id), options=options.public_options())


# ---------- results: the history view and the HTML backup (static/js/report.js) ----------

@bp.get('/report.json')
@teacher_required
def report_json():
    """?student=<id>: one student; ?teacher=<id> (the system admin): that teacher's students; ?class=1: the
    students "delete all students" deletes (own_class); else all of mine."""
    teacher = g.user
    if (student_id := request.args.get('student', type=int)) is not None:
        students = [get_student(student_id)]
    elif request.args.get('class'):
        students = own_class(g.user)
    elif (teacher_id := request.args.get('teacher', type=int)) is not None:
        if not is_system_admin(g.user):
            abort(403)
        teacher = db.get_or_404(User, teacher_id)
        if not teacher.is_teacher:
            abort(404)
        students = db.session.scalars(students_of(teacher)).all()
    else:
        students = my_students()
    data = report.build(students)
    data['teacher'] = {'name': teacher.name, 'prefix': teacher.prefix}
    resp = jsonify(data)
    resp.headers['X-Report-Size'] = str(len(resp.get_data()))   # for the backup's progress bar: Content-Length goes
    return resp                                                  # when Cloudflare compresses, this header stays


@bp.get('/report-bundle.js')
@teacher_required
def report_bundle():
    return Response(bundle(REPORT_MODULES), mimetype='text/javascript')


# ---------- teacher accounts (the system admin) ----------

@bp.post('/teachers')
@system_admin_required
def add_teacher():
    f = request.form
    login_id, name = f.get('login_id', '').strip(), f.get('name', '').strip()
    prefix, email, memo = f.get('prefix', '').strip().lower(), f.get('email', '').strip(), f.get('memo', '').strip()
    if not login_id or not name:
        flash(_('Enter a login ID and a name.'), 'error')
    elif prefix and not PREFIX.fullmatch(prefix):
        flash(_('A prefix is 2 to 10 English letters.'), 'error')
    elif db.session.scalar(db.select(User.id).filter_by(login_id=login_id)) is not None:
        flash(_('The login ID %(login_id)s is already in use.', login_id=login_id), 'error')
    else:
        # no password: set at the first login
        db.session.add(User(login_id=login_id, name=name, role='teacher', prefix=prefix,
                            email=email or None, memo=memo or None))
        db.session.commit()
        flash(_('Teacher %(name)s added. The first login sets the password.', name=name), 'info')
    return redirect(url_for('.index'))


def get_other_teacher(user_id):
    user = db.get_or_404(User, user_id)
    if not user.is_teacher or user == g.user:          # the system admin keeps its account and password
        abort(404)
    return user


@bp.post('/teachers/<int:user_id>/edit')
@system_admin_required
def edit_teacher(user_id):
    """The e-mail and the memo (who this is, what was agreed); the prefix stays, it is in the students' login IDs."""
    user = get_other_teacher(user_id)
    user.email = request.form.get('email', '').strip() or None
    user.memo = request.form.get('memo', '').strip() or None
    db.session.commit()
    flash(_('Saved the details of %(name)s.', name=user.name), 'info')
    return redirect(url_for('.index'))


@bp.post('/teachers/<int:user_id>/advanced')
@system_admin_required
def toggle_advanced(user_id):
    user = get_other_teacher(user_id)
    if not user.prefix:                                 # the home school always has them
        abort(404)
    user.advanced = not user.advanced
    db.session.commit()
    flash(_('Advanced features are on for %(name)s and their students.', name=user.name) if user.advanced
          else _('Advanced features are off for %(name)s and their students.', name=user.name), 'info')
    return redirect(url_for('.index'))


@bp.post('/teachers/<int:user_id>/reset-password')
@system_admin_required
def reset_teacher_password(user_id):
    user = get_other_teacher(user_id)
    user.password_hash = None
    db.session.commit()
    flash(_('Password reset for the teacher %(name)s. The next login sets a new one.', name=user.name), 'info')
    return redirect(url_for('.index'))


@bp.post('/teachers/<int:user_id>/delete')
@system_admin_required
def delete_teacher(user_id):
    """With their own runs and the students only they have (a prefixed teacher's class, at the end of the term)."""
    user = get_other_teacher(user_id)
    students = only_theirs(user)
    for s in students:
        db.session.delete(s)
    db.session.delete(user)
    db.session.commit()
    flash(_('Deleted %(name)s and their history.', name=user.name) if not students else
          _('Deleted %(name)s and %(n)d students with their history.', name=user.name, n=len(students)), 'info')
    return redirect(url_for('.index'))
