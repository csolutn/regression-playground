"""① Training data: generate from a formula or read a CSV, split, and standardize."""
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import expr

# Preset target functions. ``expr`` uses the same syntax students type for a custom formula.
PRESETS_1D = {
    'abs':    {'expr': 'abs(x)'},
    'xsin':   {'expr': 'x * sin(x)'},
    'square': {'expr': '0.2 * x**2'},
    'cubic':  {'expr': '0.05 * x**3 - x'},
    'sin':    {'expr': 'sin(x)'},
    'step':   {'expr': 'sign(x)'},
    'relu':   {'expr': 'max(x, 0)'},
    'wave':   {'expr': 'sin(2*x) + 0.3*x'},
    'bump':   {'expr': 'exp(-x**2)'},
}
PRESETS_2D = {
    'sin_bowl': {'expr': 'sin(x1) + 0.1 * x2**2'},
    'plane':    {'expr': '0.5*x1 - 0.3*x2'},
    'bowl':     {'expr': '0.1 * (x1**2 + x2**2)'},
    'saddle':   {'expr': '0.1 * (x1**2 - x2**2)'},
    'ripple':   {'expr': 'sin(sqrt(x1**2 + x2**2))'},
    'waves':    {'expr': 'sin(x1) * cos(x2)'},
}

PLOT_POINTS_1D = 200   # points on the prediction curve
PLOT_GRID_2D = 30      # grid × grid points on the prediction surface
DATA_VERSION = 1       # raise it when the data a config builds changes: browser runs are replayed from their settings


class DataError(ValueError):
    """Carries an English msgid plus params, translated by the API layer."""

    def __init__(self, msgid, **params):
        super().__init__(msgid)
        self.msgid, self.params = msgid, params


@dataclass
class Dataset:
    feature_names: list
    target_name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_plot: np.ndarray           # inputs where the model's prediction is drawn
    y_plot_true: np.ndarray | None  # true function on X_plot (formula data only)
    axes: list                   # 1 input: [x values]; 2 inputs: [x1 grid axis, x2 grid axis]

    @property
    def n_inputs(self):
        return self.X_train.shape[1]

    # Standardization uses training-data statistics, as in the notebook, so losses are comparable.
    def __post_init__(self):
        self.x_mean = self.X_train.mean(0)
        self.x_std = np.maximum(self.X_train.std(0), 1e-12)
        self.y_mean = self.y_train.mean()
        self.y_std = max(self.y_train.std(), 1e-12)

    def scale_x(self, a):
        return (a - self.x_mean) / self.x_std

    def scale_y(self, a):
        return (a - self.y_mean) / self.y_std

    def unscale_y(self, a):
        return a * self.y_std + self.y_mean


def expression_for(cfg):
    n = cfg['n_inputs']
    key, presets = (cfg['function_1d'], PRESETS_1D) if n == 1 else (cfg['function_2d'], PRESETS_2D)
    if key == 'custom':
        return cfg['expression_1d'] if n == 1 else cfg['expression_2d']
    return presets[key]['expr']


def target(cfg, X):
    names = ['x'] if X.shape[1] == 1 else ['x1', 'x2']
    try:
        return expr.evaluate(expression_for(cfg), {n: X[:, i] for i, n in enumerate(names)})
    except expr.ExpressionError as exc:
        vars_ = ', '.join(names)
        if exc.kind == 'syntax':
            raise DataError('The formula has a syntax error.') from exc
        if exc.kind == 'unknown_name':
            raise DataError('Unknown name in the formula: {name}. Use {vars}.', name=exc.name, vars=vars_) from exc
        if exc.kind == 'not_finite':
            raise DataError('The formula gives invalid values (for example division by zero '
                            'or the log of a negative number).') from exc
        raise DataError('The formula uses something that is not allowed. Use {vars}, numbers, + - * / ** '
                        'and functions like sin, cos, exp, abs, sqrt, max.', vars=vars_) from exc


def read_csv(cfg):
    try:
        df = pd.read_csv(io.StringIO(cfg['csv_text']))
    except Exception as exc:
        raise DataError('The CSV file could not be read.') from exc
    cols = cfg['csv_features'] + [cfg['csv_target']]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataError('Column not found in the CSV: {columns}', columns=', '.join(missing))
    df = df[cols].apply(pd.to_numeric, errors='coerce').dropna()
    if len(df) < 10:
        raise DataError('The CSV needs at least 10 rows of numbers.')
    df = df.iloc[:5000]
    return df[cfg['csv_features']].to_numpy(float), df[cfg['csv_target']].to_numpy(float)


def build_dataset(cfg):
    rng = np.random.default_rng(cfg['seed'])
    n_in = cfg['n_inputs']

    if cfg['data_source'] == 'function':
        lo, hi = np.full(n_in, cfg['x_min']), np.full(n_in, cfg['x_max'])
        if n_in == 1:
            X = np.linspace(lo[0], hi[0], cfg['data_size']).reshape(-1, 1)
        else:
            X = rng.uniform(lo[0], hi[0], size=(cfg['data_size'], 2))
        y = target(cfg, X) + rng.normal(0, cfg['noise_std'], len(X))
        feature_names, target_name = (['x'] if n_in == 1 else ['x1', 'x2']), 'y'
    else:
        X, y = read_csv(cfg)
        lo, hi = X.min(0), X.max(0)
        feature_names, target_name = list(cfg['csv_features']), cfg['csv_target']

    idx = rng.permutation(len(X))
    n_val = max(1, int(len(X) * cfg['validation_ratio']))
    val, train = idx[:n_val], idx[n_val:]

    if n_in == 1:
        axes = [np.linspace(lo[0], hi[0], PLOT_POINTS_1D)]
        X_plot = axes[0].reshape(-1, 1)
    else:
        axes = [np.linspace(lo[i], hi[i], PLOT_GRID_2D) for i in range(2)]
        G1, G2 = np.meshgrid(*axes)          # row = x2 index, column = x1 index
        X_plot = np.column_stack([G1.ravel(), G2.ravel()])
    y_true = target(cfg, X_plot) if cfg['data_source'] == 'function' else None

    return Dataset(feature_names, target_name, X[train], y[train], X[val], y[val], X_plot, y_true, axes)
