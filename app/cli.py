"""Command-line setup:  uv run flask create-teacher  /  uv run flask import-roster roster.csv"""
import click
from flask import Blueprint

from .extensions import db
from .models import User
from .roster import import_roster, parse_roster

bp = Blueprint('cli', __name__, cli_group=None)


@bp.cli.command('create-teacher')
@click.option('--login-id', prompt='Teacher login ID')
@click.option('--name', prompt='Teacher name')
@click.password_option()
def create_teacher(login_id, name, password):
    """Create a teacher account (or reset its password)."""
    user = db.session.scalar(db.select(User).filter_by(login_id=login_id))
    if user is None:
        user = User(login_id=login_id, name=name, role='teacher')
        db.session.add(user)
    user.name, user.role = name, 'teacher'
    user.set_password(password)
    db.session.commit()
    click.echo(f'Teacher account ready: {login_id} ({name})')


@bp.cli.command('import-roster')
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
def import_roster_command(path):
    """Import a class list CSV (student_id,name or 학번,이름)."""
    with open(path, encoding='utf-8-sig') as f:
        added, updated = import_roster(parse_roster(f.read()))
    click.echo(f'{added} added, {updated} updated')
