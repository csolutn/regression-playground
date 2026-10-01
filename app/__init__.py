from dotenv import load_dotenv

load_dotenv()   # before Config reads the environment

from flask import Flask  # noqa: E402

from .config import Config  # noqa: E402
from .extensions import babel, db  # noqa: E402
from .i18n import LANGUAGES, js_catalog, select_locale  # noqa: E402


def create_app(config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config)
    if config:
        app.config.update(config)
    if not (app.debug or app.testing) and app.config['SECRET_KEY'] in ('dev', 'change-me', ''):
        raise RuntimeError('Set SECRET_KEY to a long random value (see deploy.env.example).')
    if app.config['TRUST_PROXY']:
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    db.init_app(app)
    babel.init_app(app, locale_selector=select_locale)

    from . import admin, api, auth, cli, views
    for module in (auth, views, api, admin, cli):
        app.register_blueprint(module.bp)

    @app.context_processor
    def inject_i18n():
        return {'languages': LANGUAGES, 'current_language': select_locale(), 'js_catalog': js_catalog}

    with app.app_context():
        from . import models  # noqa: F401  (register tables)
        db.create_all()
    return app
