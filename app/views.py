from flask import Blueprint, g, render_template

from .auth import user_or_guest_required
from .ml import options

bp = Blueprint('main', __name__)


@bp.get('/')
@user_or_guest_required
def index():
    # a school with a prefix and advanced features off: the tree models are not shown at all
    advanced = g.user is None or g.user.can_use_advanced
    return render_template('index.html', options={**options.public_options(), 'advanced': advanced}, advanced=advanced)
