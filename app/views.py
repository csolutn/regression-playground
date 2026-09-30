from flask import Blueprint, render_template

from .auth import login_required
from .ml import options

bp = Blueprint('main', __name__)


@bp.get('/')
@login_required
def index():
    return render_template('index.html', options=options.public_options())
