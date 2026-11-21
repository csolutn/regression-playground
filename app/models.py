"""Database tables: users (roster) and training runs (history)."""
import gzip
import json
from datetime import datetime, timezone

from sqlalchemy import Column, ForeignKey, LargeBinary, String, Table, Text, inspect, text
from sqlalchemy.orm import Mapped, deferred, mapped_column, relationship
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


# which teacher put which student on their class list (a teacher with a prefix sees only these)
teacher_student = Table(
    'teacher_student', db.metadata,
    Column('teacher_id', ForeignKey('user.id'), primary_key=True),
    Column('student_id', ForeignKey('user.id'), primary_key=True, index=True),
)


class User(db.Model):
    """A student from the teacher's roster, or a teacher.

    ``password_hash`` is empty until the first login, which sets the password.
    ``prefix``: another school's letters (e.g. 'abc'), set by the system admin on a teacher; that teacher's students
    get it in front of their student number (login ID 'abc21013') and the teacher sees only them. Without a prefix
    (the home school) students log in with the bare number. A prefixed school uses no advanced features (server
    training: tree models and runs too slow for the browser) unless the system admin turns them on for its teacher.
    """
    id: Mapped[int] = mapped_column(primary_key=True)
    login_id: Mapped[str] = mapped_column(String(40), unique=True, index=True)   # student number
    name: Mapped[str] = mapped_column(String(80))
    role: Mapped[str] = mapped_column(String(10), default='student')            # student | teacher
    password_hash: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    last_login_at: Mapped[datetime | None]
    prefix: Mapped[str] = mapped_column(String(10), default='', server_default='')
    email: Mapped[str | None] = mapped_column(String(120))         # a teacher's, for the system admin's records
    memo: Mapped[str | None] = mapped_column(Text)                 # the system admin's note on a teacher
    advanced: Mapped[bool] = mapped_column(default=False, server_default='0')   # a prefixed teacher's switch

    runs: Mapped[list['Run']] = relationship(back_populates='user', cascade='all, delete-orphan',
                                             order_by='Run.id')
    students: Mapped[list['User']] = relationship(
        secondary=teacher_student, back_populates='teachers',
        primaryjoin=lambda: User.id == teacher_student.c.teacher_id,
        secondaryjoin=lambda: User.id == teacher_student.c.student_id)
    teachers: Mapped[list['User']] = relationship(
        secondary=teacher_student, back_populates='students',
        primaryjoin=lambda: User.id == teacher_student.c.student_id,
        secondaryjoin=lambda: User.id == teacher_student.c.teacher_id)

    @property
    def is_teacher(self):
        return self.role == 'teacher'

    @property
    def number(self):
        """The student number without the prefix (what the teacher typed)."""
        return self.login_id[len(self.prefix):] if self.prefix and self.login_id.startswith(self.prefix) else self.login_id

    @property
    def can_use_advanced(self):
        if not self.prefix:
            return True
        if self.is_teacher:
            return self.advanced
        return any(t.advanced for t in self.teachers)

    @property
    def has_password(self):
        return bool(self.password_hash)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return bool(self.password_hash) and check_password_hash(self.password_hash, password)


# columns added to existing tables since the first release (create_all only makes missing tables)
ADDED_COLUMNS = {
    'user': [('prefix', "VARCHAR(10) NOT NULL DEFAULT ''"), ('email', 'VARCHAR(120)'), ('memo', 'TEXT'),
             ('advanced', 'BOOLEAN NOT NULL DEFAULT 0')],
}


def add_missing_columns():
    for table, columns in ADDED_COLUMNS.items():
        have = {c['name'] for c in inspect(db.engine).get_columns(table)}
        for name, ddl in columns:
            if name not in have:
                db.session.execute(text(f'ALTER TABLE "{table}" ADD COLUMN {name} {ddl}'))
    db.session.commit()


class Run(db.Model):
    """One training run: its settings, final losses, and the stored animation (gzip JSON)."""
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('user.id'), index=True)
    seq: Mapped[int]                       # 1, 2, 3 … per user
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    model: Mapped[str] = mapped_column(String(30))
    status: Mapped[str] = mapped_column(String(20))
    final_train: Mapped[float | None]
    final_val: Mapped[float | None]
    steps: Mapped[int]
    duration: Mapped[float]
    config_json: Mapped[str] = mapped_column(Text)
    payload: Mapped[bytes] = deferred(mapped_column(LargeBinary))

    user: Mapped[User] = relationship(back_populates='runs')

    @property
    def config(self):
        return json.loads(self.config_json)

    def set_payload(self, obj):
        # level 6, not gzip's 9: a 2-input run's frames take 0.06 s instead of 0.5 s, for 8 % more bytes
        self.payload = gzip.compress(json.dumps(obj, separators=(',', ':')).encode(), compresslevel=6)

    def get_payload(self):
        return json.loads(gzip.decompress(self.payload))

    def to_row(self, cfg=None):
        """Summary for the history table (the CSV text itself is left out); ``cfg``: the settings, if read already."""
        cfg = self.config if cfg is None else cfg
        cfg.pop('csv_text', None)
        return {
            'id': self.id, 'seq': self.seq, 'created_at': self.created_at.isoformat() + 'Z',
            'model': self.model, 'status': self.status, 'final_train': self.final_train,
            'final_val': self.final_val, 'steps': self.steps, 'duration': self.duration, 'config': cfg,
        }
