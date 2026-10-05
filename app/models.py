"""Database tables: users (roster) and training runs (history)."""
import gzip
import json
from datetime import datetime, timezone

from sqlalchemy import ForeignKey, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, deferred, mapped_column, relationship
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(db.Model):
    """A student from the teacher's roster, or a teacher.

    ``password_hash`` is empty until the first login, which sets the password.
    """
    id: Mapped[int] = mapped_column(primary_key=True)
    login_id: Mapped[str] = mapped_column(String(40), unique=True, index=True)   # student number
    name: Mapped[str] = mapped_column(String(80))
    role: Mapped[str] = mapped_column(String(10), default='student')            # student | teacher
    password_hash: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    last_login_at: Mapped[datetime | None]

    runs: Mapped[list['Run']] = relationship(back_populates='user', cascade='all, delete-orphan',
                                             order_by='Run.id')

    @property
    def is_teacher(self):
        return self.role == 'teacher'

    @property
    def has_password(self):
        return bool(self.password_hash)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return bool(self.password_hash) and check_password_hash(self.password_hash, password)


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
