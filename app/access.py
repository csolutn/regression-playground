"""Which students a teacher may see.

The system admin (the oldest teacher account) sees every student. A teacher without a prefix (the home school)
sees the students without one, as before prefixes existed. A teacher with a prefix sees only the students on
their own class list (teacher_student).
"""
from .extensions import db
from .models import User, teacher_student


def system_admin_id():
    return db.session.scalar(db.select(User.id).filter_by(role='teacher').order_by(User.id).limit(1))


def is_system_admin(user):
    return user is not None and user.is_teacher and user.id == system_admin_id()


def students_of(teacher):
    """A select() of the students this teacher may see, by prefix and student number."""
    query = db.select(User).filter_by(role='student').order_by(User.prefix, User.login_id)
    if is_system_admin(teacher):
        return query
    if not teacher.prefix:
        return query.filter(User.prefix == '')
    mine = db.select(teacher_student.c.student_id).where(teacher_student.c.teacher_id == teacher.id)
    return query.filter(User.id.in_(mine))


def can_see(teacher, user):
    """A teacher may see their own account and the students of students_of()."""
    if user.id == teacher.id:
        return True
    if not teacher.is_teacher or user.is_teacher:
        return False
    return db.session.scalar(students_of(teacher).with_only_columns(User.id).filter(User.id == user.id)) is not None
