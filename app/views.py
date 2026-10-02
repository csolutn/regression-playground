from flask import Blueprint, render_template

from .auth import user_or_guest_required
from .ml import options

bp = Blueprint('main', __name__)


@bp.get('/')
@user_or_guest_required
def index():
    return render_template('index.html', options=options.public_options())
