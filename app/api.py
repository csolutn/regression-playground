"""JSON API used by the playground page.

POST /api/train streams NDJSON events from app.ml.trainer (see its docstring), then
saves the run and sends {'type': 'saved', 'run': row}. The browser draws everything,
so the server only sends numbers.
"""
import json
import threading

from flask import Blueprint, Response, abort, current_app, g, jsonify, request, stream_with_context
from flask_babel import gettext

from .auth import login_required
from .extensions import db
from .ml import data as ml_data
from .ml import options, trainer
from .models import Run, User

bp = Blueprint('api', __name__, url_prefix='/api')
_slots = None
_slots_lock = threading.Lock()


def training_slots():
    """Limits how many runs train at the same time (MAX_CONCURRENT_TRAININGS)."""
    global _slots
    with _slots_lock:
        if _slots is None:
            _slots = threading.BoundedSemaphore(current_app.config['MAX_CONCURRENT_TRAININGS'])
        return _slots


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


@bp.post('/train')
@login_required
def train():
    cfg, errors = read_config()
    if errors:
        return error_response(errors)
    try:                           # fail fast on bad data before starting the stream
        events = trainer.train(cfg)
        first = next(events)
    except (ml_data.DataError, trainer.BudgetError) as exc:
        return error_response([(exc.msgid, exc.params)])
    user_id = g.user.id

    @stream_with_context
    def generate():
        slots = training_slots()
        if not slots.acquire(blocking=False):
            yield line({'type': 'queued'})
            slots.acquire()
        try:
            stored = {'meta': first['meta'], 'frames': [], 'loss': {'steps': [], 'train': [], 'val': []}}
            yield line(first)
            end = None
            for event in events:
                if event['type'] == 'frame':
                    stored['frames'].append({k: event[k] for k in ('step', 'pred', 'train', 'val')})
                elif event['type'] == 'loss':
                    for k in stored['loss']:
                        stored['loss'][k] += event[k]
                elif event['type'] == 'end':
                    end = event
                yield line(event)
            run = save_run(user_id, cfg, end, stored)
            yield line({'type': 'saved', 'run': run.to_row()})
        finally:
            slots.release()

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
