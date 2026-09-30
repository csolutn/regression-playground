"""Class list (roster) import, shared by the teacher page and the CLI."""
import csv
import io

from .extensions import db
from .models import User

ID_HEADERS = {'student_id', 'id', 'login_id', '학번'}
NAME_HEADERS = {'name', '이름', '성명'}


def parse_roster(text):
    """Rows of (student number, name). A header row is optional; without one the first two columns are used."""
    rows = [r for r in csv.reader(io.StringIO(text.lstrip('﻿'))) if any(c.strip() for c in r)]
    if not rows:
        return []
    header = [c.strip().lower() for c in rows[0]]
    id_col = next((i for i, h in enumerate(header) if h in ID_HEADERS), None)
    name_col = next((i for i, h in enumerate(header) if h in NAME_HEADERS), None)
    if id_col is not None and name_col is not None:
        rows = rows[1:]
    else:
        id_col, name_col = 0, 1
    return [(r[id_col].strip(), r[name_col].strip()) for r in rows
            if len(r) > max(id_col, name_col) and r[id_col].strip() and r[name_col].strip()]


def import_roster(pairs):
    """Adds new students and updates names of existing ones. Returns (added, updated)."""
    added = updated = 0
    for login_id, name in pairs:
        user = db.session.scalar(db.select(User).filter_by(login_id=login_id))
        if user is None:
            db.session.add(User(login_id=login_id, name=name, role='student'))
            added += 1
        elif user.name != name and not user.is_teacher:
            user.name = name
            updated += 1
    db.session.commit()
    return added, updated
