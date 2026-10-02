import hashlib
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()   # before Config reads the environment

from flask import Flask, request, send_from_directory  # noqa: E402

from .config import Config  # noqa: E402
from .extensions import babel, db  # noqa: E402
from .i18n import LANGUAGES, js_catalog, select_locale  # noqa: E402


STATIC_DIR = Path(__file__).parent / 'static'


def static_version():
    """A hash of every file under static/. It is part of each static URL (/static/<version>/js/...),
    so after a deploy browsers fetch the new files instead of keeping old ones from their cache."""
    h = hashlib.sha1()
    for p in sorted(STATIC_DIR.rglob('*')):
        if p.is_file():
            h.update(str(p.relative_to(STATIC_DIR)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:10]


def create_app(config=None):
    version = static_version()
    app = Flask(__name__, instance_relative_config=True, static_url_path=f'/static/{version}')
    app.config.from_object(Config)
    if config:
        app.config.update(config)
    app.config['STATIC_VERSION'] = version
    # a versioned URL never changes its content: cache it for a year (in debug, check every time)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = None if app.debug else 365 * 24 * 3600

    @app.get('/static/<path:filename>')
    def static_unversioned(filename):
        """Pages opened before a deploy still ask for unversioned or older-version URLs (a module they
        import later, the training worker): serve today's file, without caching."""
        head, _, rest = filename.partition('/')
        if rest and len(head) == 10 and all(c in '0123456789abcdef' for c in head):
            filename = rest
        return send_from_directory(app.static_folder, filename, max_age=0)

    @app.get('/favicon.ico')
    @app.get('/apple-touch-icon.png')
    def root_icon():
        """Browsers and iOS ask for these at the site root, whatever the page links."""
        return send_from_directory(STATIC_DIR / 'icons', request.path.lstrip('/'), max_age=24 * 3600)
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
