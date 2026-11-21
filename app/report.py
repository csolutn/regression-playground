"""The data of the teacher's history view and of the HTML backup (static/js/report.js draws it).

For each student their runs, newest first, each with the data it learned from (a meta, shared by the runs on the
same data) and, for a run trained on the server, its last frame and loss curve. A run trained in the browser has
only its results stored: the teacher's browser trains it again from its settings (train.js replayRun) to draw it.
Nothing of this is stored: it is made when it is asked for.
"""
import json
import math

from .extensions import db
from .ml import data as ml_data
from .ml import trainer
from .models import Run

RUN_META_KEYS = ('structure', 'model', 'loss', 'total_steps')       # the rest of a meta is the data
# settings that do not change the data: left out of the key that shares a dataset between runs
# (one missing here only builds the same data twice)
MODEL_KEYS = ('model', 'hidden_layers', 'activation', 'max_depth', 'n_estimators', 'loss', 'huber_delta',
              'optimizer', 'learning_rate', 'epochs', 'batch_method', 'batch_size', 'tree_learning_rate')
MAX_LOSS_POINTS = 200


class Metas:
    """Each distinct data meta once; runs refer to it by its index."""
    def __init__(self):
        self.list, self.index, self.by_settings = [], {}, {}

    def add(self, meta):
        data = {k: v for k, v in meta.items() if k not in RUN_META_KEYS}
        key = json.dumps(data, sort_keys=True, separators=(',', ':'))
        if key not in self.index:
            self.index[key] = len(self.list)
            self.list.append(data)
        return self.index[key]

    def for_settings(self, cfg):
        """The data a browser run's settings build now (as POST /api/preview would send)."""
        key = json.dumps({k: v for k, v in cfg.items() if k not in MODEL_KEYS}, sort_keys=True)
        if key not in self.by_settings:
            self.by_settings[key] = self.add(trainer.meta(cfg, ml_data.build_dataset(cfg)))
        return self.by_settings[key]


def thin(loss, n=MAX_LOSS_POINTS):
    """At most about n points of the loss curve, always with the last one."""
    total = len(loss['steps'])
    k = max(1, math.ceil(total / n))
    keep = list(range(0, total, k))
    if total and keep[-1] != total - 1:
        keep.append(total - 1)
    return {s: [loss[s][i] for i in keep] for s in ('steps', 'train', 'val')}


def run_entry(run, metas):
    cfg = run.config
    entry = {'row': run.to_row(dict(cfg))}
    payload = run.get_payload()
    try:
        if 'replay' in payload:
            entry['meta'] = metas.for_settings(cfg)
            entry['replay'] = {'trainer': payload['replay']['trainer'],
                               'data_changed': payload['replay']['data'] != ml_data.DATA_VERSION}
            entry['total_steps'] = trainer.total_steps(cfg)
        else:
            entry['meta'] = metas.add(payload['meta'])
            entry['total_steps'] = payload['meta']['total_steps']
            frames = payload['frames']
            entry['result'] = {'frame': frames[-1] if frames else None, 'loss': thin(payload['loss'])}
    except ml_data.DataError:                 # its data can not be built any more
        entry.pop('meta', None)
        entry['error'] = True
    return entry


def build(students):
    metas = Metas()
    out = []
    for s in students:
        runs = db.session.scalars(db.select(Run).filter_by(user_id=s.id).order_by(Run.id.desc())).all()
        out.append({'id': s.id, 'prefix': s.prefix, 'number': s.number, 'name': s.name,
                    'runs': [run_entry(r, metas) for r in runs]})
    return {'students': out, 'metas': metas.list}
