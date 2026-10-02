"""JSON API used by the playground page.

Linear and neural network runs usually train in the browser: POST /api/prepare returns the data
(the trainer's meta) and whether the browser should train it (options.trains_in_browser); the
browser then saves the run with POST /api/runs.

POST /api/train trains on the server instead (tree models, and the runs the browser would be
slow at). It streams NDJSON events from app.ml.trainer (see its docstring), then saves the run
and sends {'type': 'saved', 'run': row}. The browser draws everything, so the server only sends
numbers. Runs train in worker processes (app.ml.pool); while all are busy the stream sends
{'type': 'queued'} every few seconds, and {'type': 'error', 'message'} when the wait or the
worker takes too long.
"""
import json
import math
import threading
import time

from flask import Blueprint, Response, abort, current_app, g, jsonify, request, stream_with_context
from flask_babel import gettext

from .auth import login_required
from .extensions import db
from .ml import data as ml_data
from .ml import options, trainer
from .ml.pool import TrainingPool
from .models import Run, User

bp = Blueprint('api', __name__, url_prefix='/api')
_pool = None
_pool_lock = threading.Lock()
QUEUE_POLL_S = 3       # a 'queued' line this often keeps the browser informed and notices one that left


def training_pool():
    """MAX_CONCURRENT_TRAININGS worker processes, started on the first run."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = TrainingPool(current_app.config['MAX_CONCURRENT_TRAININGS'])
        return _pool


def translate(msgid, params):
    """Messages from app.ml use {name} placeholders; a {field} is shown by its label."""
    if 'field' in params:
        params = {**params, 'field': gettext(options.FIELD_LABELS.get(params['field'], params['field']))}
    return gettext(msgid).format(**params)


def error_response(errors, status=400):
    return jsonify({'errors': [translate(msgid, params) for msgid, params in errors]}), status


def read_config():
    if not request.is_json:        # also blocks cross-site form posts
        abort(415)
    cfg, errors = options.validate(request.get_json(silent=True))
    return cfg, errors


def line(obj):
    return json.dumps(obj, separators=(',', ':'), allow_nan=False) + '\n'


@bp.post('/preview')
@login_required
def preview():
    """Data points and the true function, for the preview in ① training data."""
    cfg, errors = read_config()
    if errors:
        return error_response(errors)
    try:
        ds = ml_data.build_dataset(cfg)
    except ml_data.DataError as exc:
        return error_response([(exc.msgid, exc.params)])
    return jsonify(trainer.meta(cfg, ds))


@bp.post('/prepare')
@login_required
def prepare():
    """For a linear or neural network run: {'browser': True, 'meta', 'time_limit_s'}, or {'browser': False}."""
    cfg, errors = read_config()
    if errors:
        return error_response(errors)
    if cfg['model'] not in options.GRADIENT_MODELS:
        abort(400)
    try:
        ds = trainer.check(cfg)
    except (ml_data.DataError, trainer.BudgetError) as exc:
        return error_response([(exc.msgid, exc.params)])
    if not options.trains_in_browser(cfg, len(ds.X_train), len(ds.X_val)):
        return jsonify({'browser': False})
    return jsonify({'browser': True, 'meta': trainer.meta(cfg, ds),
                    'time_limit_s': current_app.config['TRAINING_TIME_LIMIT_S']})


@bp.post('/train')
@login_required
def train():
    cfg, errors = read_config()
    if errors:
        return error_response(errors)
    try:                           # fail fast on bad data before starting the stream
        trainer.check(cfg)
    except (ml_data.DataError, trainer.BudgetError) as exc:
        return error_response([(exc.msgid, exc.params)])
    user_id, conf = g.user.id, current_app.config

    @stream_with_context
    def generate():
        pool = training_pool()
        if not pool.acquire(timeout=0):
            give_up = time.monotonic() + conf['QUEUE_TIMEOUT_S']
            while True:
                yield line({'type': 'queued'})
                if pool.acquire(timeout=QUEUE_POLL_S):
                    break
                if time.monotonic() > give_up:
                    yield line({'type': 'error', 'message': gettext('The server is busy. Try again in a moment.')})
                    return
        limit = conf['TRAINING_TIME_LIMIT_S']
        stored = {'meta': None, 'frames': [], 'loss': {'steps': [], 'train': [], 'val': []}}
        end = None
        try:
            for event in pool.stream(cfg, time_limit=limit, idle_timeout=limit + 30):   # frees the slot itself
                if event['type'] == 'start':
                    stored['meta'] = event['meta']
                elif event['type'] == 'frame':
                    stored['frames'].append({k: event[k] for k in ('step', 'pred', 'train', 'val')})
                elif event['type'] == 'loss':
                    for k in stored['loss']:
                        stored['loss'][k] += event[k]
                elif event['type'] == 'end':
                    end = event
                yield line(event)
        except (RuntimeError, TimeoutError):
            current_app.logger.exception('training failed')
            yield line({'type': 'error', 'message': gettext('Training failed on the server. Try again.')})
            return
        run = save_run(user_id, cfg, end, stored)
        yield line({'type': 'saved', 'run': run.to_row()})

    return Response(generate(), mimetype='application/x-ndjson',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


def save_run(user_id, cfg, end, stored):
    last_seq = db.session.scalar(db.select(db.func.max(Run.seq)).filter_by(user_id=user_id)) or 0
    run = Run(user_id=user_id, seq=last_seq + 1, model=cfg['model'], status=end['status'],
              final_train=end['final_train'], final_val=end['final_val'], steps=end['steps'],
              duration=end['duration'], config_json=json.dumps(cfg, ensure_ascii=False))
    run.set_payload(stored)
    db.session.add(run)
    db.session.commit()
    return run


MAX_UPLOAD_BYTES = 8 * 1024 * 1024     # 300 frames of a 30 × 30 surface are about 2 MB


@bp.post('/runs')
@login_required
def upload_run():
    """Save a run trained in the browser: {config, frames, loss, end}. The server rebuilds the data itself."""
    request.max_content_length = MAX_UPLOAD_BYTES
    if not request.is_json:
        abort(415)
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        abort(400)
    cfg, errors = options.validate(body.get('config'))
    if errors:
        return error_response(errors)
    if cfg['model'] not in options.GRADIENT_MODELS:
        abort(400)
    try:
        ds = trainer.check(cfg)
    except (ml_data.DataError, trainer.BudgetError) as exc:
        return error_response([(exc.msgid, exc.params)])
    try:
        frames, loss, end = checked_upload(body, cfg['epochs'], len(ds.X_plot))
    except (KeyError, TypeError, ValueError):
        abort(400)
    run = save_run(g.user.id, cfg, end, {'meta': trainer.meta(cfg, ds), 'frames': frames, 'loss': loss})
    return jsonify({'run': run.to_row()}), 201


def checked_upload(body, epochs, n_plot):
    """The uploaded frames, loss history and end, in the trainer's shapes; raises on anything else."""
    def number(v, optional=False):
        if v is None and optional:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(v)
        return v

    def step(v):
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= epochs:
            raise ValueError(v)
        return v

    frames = [{'step': step(f['step']), 'pred': [number(v) for v in f['pred']],
               'train': number(f['train'], True), 'val': number(f['val'], True)} for f in body['frames']]
    if not 1 <= len(frames) <= trainer.MAX_FRAMES + 2 or any(len(f['pred']) != n_plot for f in frames):
        raise ValueError('frames')
    loss = {'steps': [step(v) for v in body['loss']['steps']],
            'train': [number(v, True) for v in body['loss']['train']],
            'val': [number(v, True) for v in body['loss']['val']]}
    if not 1 <= len(loss['steps']) == len(loss['train']) == len(loss['val']) <= epochs:
        raise ValueError('loss')
    e = body['end']
    if e['status'] not in ('done', 'timeout', 'diverged'):
        raise ValueError(e['status'])
    end = {'status': e['status'], 'steps': step(e['steps']), 'final_train': number(e['final_train'], True),
           'final_val': number(e['final_val'], True), 'duration': number(e['duration'])}
    return frames, loss, end


def visible_user_id():
    """Students see their own runs; teachers may pass ?user=<id>."""
    uid = request.args.get('user', type=int)
    if uid and uid != g.user.id:
        if not g.user.is_teacher or db.session.get(User, uid) is None:
            abort(403)
        return uid
    return g.user.id


def get_run(run_id):
    run = db.get_or_404(Run, run_id)
    if run.user_id != g.user.id and not g.user.is_teacher:
        abort(404)
    return run


@bp.get('/runs')
@login_required
def list_runs():
    runs = db.session.scalars(db.select(Run).filter_by(user_id=visible_user_id()).order_by(Run.id.desc()))
    return jsonify([r.to_row() for r in runs])


@bp.get('/runs/<int:run_id>')
@login_required
def run_detail(run_id):
    """Everything needed to replay the animation, plus the full settings (with CSV text)."""
    run = get_run(run_id)
    return jsonify({'run': run.to_row(), 'config': run.config, **run.get_payload()})


@bp.delete('/runs/<int:run_id>')
@login_required
def delete_run(run_id):
    run = get_run(run_id)
    db.session.delete(run)
    db.session.commit()
    return '', 204
