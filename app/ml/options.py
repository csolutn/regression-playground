"""Experiment settings: defaults, choices, limits and validation.

This is the single source of truth for every setting in the UI. To add one:
  1. add it to DEFAULTS (and CHOICES or RANGES when it is constrained),
  2. add an input with the same ``name`` to the step template in app/templates/steps/,
  3. read it in data.py or trainer.py.

Error messages are English msgids (marked with N_) and are translated by the API layer.
"""
import math

from .data import PRESETS_1D, PRESETS_2D


def N_(s):
    return s


GRADIENT_MODELS = ('linear', 'neural_net')
PLAIN_GD = ('bgd', 'sgd', 'mini')     # gradient descent whose name fixes the batch: all data / 1 point / a mini-batch
TREE_MODELS = ('decision_tree', 'random_forest', 'gradient_boosting')

DEFAULTS = {
    # ① training data
    'data_source': 'function',
    'n_inputs': 1,
    'function_1d': 'abs',
    'function_2d': 'sin_bowl',
    'expression_1d': 'abs(x)',
    'expression_2d': 'sin(x1) + 0.1 * x2**2',
    'x_min': -5.0,
    'x_max': 5.0,
    'data_size': 500,
    'noise_std': 0.1,
    'validation_ratio': 0.2,
    'seed': 42,
    'csv_name': '',
    'csv_text': '',
    'csv_features': [],
    'csv_target': '',
    # ② prediction function (model)
    'model': 'neural_net',
    'hidden_layers': [4],
    'activation': 'relu',
    'max_depth': 4,
    'n_estimators': 50,
    # ③ loss function
    'loss': 'mse',
    'huber_delta': 1.0,
    # ④ optimization
    'optimizer': 'bgd',
    'learning_rate': 0.01,
    'epochs': 1000,
    'batch_method': 'bgd',
    'batch_size': 32,
    'tree_learning_rate': 0.1,
}

CHOICES = {
    'data_source': ('function', 'csv'),
    'n_inputs': (1, 2),
    'function_1d': tuple(PRESETS_1D) + ('custom',),
    'function_2d': tuple(PRESETS_2D) + ('custom',),
    'model': GRADIENT_MODELS + TREE_MODELS,
    'activation': ('relu', 'leaky_relu', 'elu', 'gelu', 'tanh', 'sigmoid', 'linear'),
    'loss': ('mse', 'mae', 'huber'),
    'optimizer': PLAIN_GD + ('momentum', 'rmsprop', 'adam'),
    'batch_method': ('sgd', 'mini', 'bgd'),
}

RANGES = {  # (min, max) inclusive
    'x_min': (-100, 100),
    'x_max': (-100, 100),
    'data_size': (20, 2000),
    'noise_std': (0, 10),
    'validation_ratio': (0.05, 0.5),
    'seed': (0, 2**31 - 1),
    'max_depth': (1, 8),
    'n_estimators': (1, 100),
    'huber_delta': (0.05, 10),
    'learning_rate': (1e-5, 1),
    'epochs': (1, 1500),
    'batch_size': (1, 2000),
    'tree_learning_rate': (0.01, 1),
}
INTS = {'n_inputs', 'data_size', 'seed', 'max_depth', 'n_estimators', 'epochs', 'batch_size'}

FIELD_LABELS = {
    'data_source': N_('Data source'), 'n_inputs': N_('Input variables'), 'function_1d': N_('Function'),
    'function_2d': N_('Function'), 'expression_1d': N_('Formula'), 'expression_2d': N_('Formula'),
    'x_min': N_('Input range (min)'), 'x_max': N_('Input range (max)'), 'data_size': N_('Number of points'),
    'noise_std': N_('Noise'), 'validation_ratio': N_('Validation ratio'), 'seed': N_('Random seed'),
    'csv_features': N_('Input columns'), 'csv_target': N_('Output column'), 'model': N_('Model'),
    'hidden_layers': N_('Hidden layers'), 'activation': N_('Activation function'), 'max_depth': N_('Max depth'),
    'n_estimators': N_('Number of trees'), 'loss': N_('Loss function'), 'huber_delta': N_('Huber δ'),
    'optimizer': N_('Optimizer'), 'learning_rate': N_('Learning rate'), 'epochs': N_('Epochs'),
    'batch_method': N_('Batch method'), 'batch_size': N_('Mini-batch size'),
    'tree_learning_rate': N_('Boosting learning rate'),
}

MAX_HIDDEN_LAYERS = 6
MAX_NEURONS = 32
MAX_GRADIENT_STEPS = 400_000     # epochs × mini-batches per epoch, protects the server; SGD on the default data at the default epochs fits
MAX_CSV_BYTES = 1_000_000
MAX_CSV_ROWS = 5000


def validate(raw):
    """Merge ``raw`` over DEFAULTS and check it. Returns (config, errors).

    Each error is (msgid, params) so the caller can translate it.
    """
    cfg, errors = dict(DEFAULTS), []
    raw = raw if isinstance(raw, dict) else {}

    for key, default in DEFAULTS.items():
        if key not in raw or raw[key] is None:
            continue
        value = raw[key]
        try:
            if isinstance(default, bool):
                value = bool(value)
            elif key in INTS:
                value = int(value)
            elif isinstance(default, float):
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError
            elif isinstance(default, str):
                value = str(value)
            elif isinstance(default, list):
                if not isinstance(value, list):
                    raise ValueError
        except (TypeError, ValueError):
            errors.append((N_('Invalid value for {field}.'), {'field': key}))
            continue
        cfg[key] = value

    for key, allowed in CHOICES.items():
        if cfg[key] not in allowed:
            errors.append((N_('Invalid value for {field}.'), {'field': key}))
    for key, (lo, hi) in RANGES.items():
        if isinstance(cfg[key], (int, float)) and not lo <= cfg[key] <= hi:
            errors.append((N_('{field} must be between {min} and {max}.'), {'field': key, 'min': lo, 'max': hi}))

    if cfg['optimizer'] in PLAIN_GD:      # e.g. SGD really steps after every single point
        cfg['batch_method'] = cfg['optimizer']

    if cfg['x_min'] >= cfg['x_max']:
        errors.append((N_('The input range minimum must be smaller than the maximum.'), {}))

    if cfg['data_source'] == 'csv':
        if not cfg['csv_text']:
            errors.append((N_('Upload a CSV file first.'), {}))
        elif len(cfg['csv_text'].encode()) > MAX_CSV_BYTES:
            errors.append((N_('The CSV file is too large (max 1 MB).'), {}))
        features = cfg['csv_features']
        if not (1 <= len(features) <= 2) or not all(isinstance(f, str) for f in features):
            errors.append((N_('Choose one or two input columns.'), {}))
        elif not cfg['csv_target'] or cfg['csv_target'] in features:
            errors.append((N_('Choose an output column that is not an input column.'), {}))
        else:
            cfg['n_inputs'] = len(features)
    else:
        cfg['csv_name'], cfg['csv_text'], cfg['csv_features'], cfg['csv_target'] = '', '', [], ''

    layers = cfg['hidden_layers']
    if not all(isinstance(n, int) and not isinstance(n, bool) for n in layers):
        errors.append((N_('Invalid value for {field}.'), {'field': 'hidden_layers'}))
    elif cfg['model'] == 'neural_net':
        if not 1 <= len(layers) <= MAX_HIDDEN_LAYERS:
            errors.append((N_('A neural network needs 1 to {max} hidden layers.'), {'max': MAX_HIDDEN_LAYERS}))
        elif not all(1 <= n <= MAX_NEURONS for n in layers):
            errors.append((N_('Each hidden layer needs 1 to {max} neurons.'), {'max': MAX_NEURONS}))

    if cfg['model'] in GRADIENT_MODELS and cfg['data_source'] == 'function' and not errors:
        # CSV row counts are only known after parsing; the trainer checks those with step_budget_error()
        n_train = cfg['data_size'] - int(cfg['data_size'] * cfg['validation_ratio'])
        if err := step_budget_error(cfg, n_train):
            errors.append(err)
    return cfg, errors


def batch_size(cfg, n_train):
    return {'sgd': 1, 'mini': cfg['batch_size'], 'bgd': n_train}[cfg['batch_method']]


def step_budget_error(cfg, n_train):
    steps = cfg['epochs'] * math.ceil(n_train / min(batch_size(cfg, n_train), n_train))
    if steps > MAX_GRADIENT_STEPS:
        return (N_('Too much computation: {steps} update steps (limit {limit}). '
                   'Reduce the epochs or use a larger batch.'),
                {'steps': f'{steps:,}', 'limit': f'{MAX_GRADIENT_STEPS:,}'})
    return None


# Linear and neural network runs train in the student's browser (app/static/js/nn.js) to spare the server,
# except when the browser would be much slower: plain JavaScript pays for every multiplication, while
# torch pays for every update step. Estimated seconds, measured on the Mac mini (Chrome's V8 via Node).
BROWSER_S_PER_MULTIPLY = 0.33e-9
BROWSER_S_PER_UPDATE = 5e-9          # per weight, per step
SERVER_S_PER_STEP = 20e-6
SERVER_S_PER_LAYER_STEP = 7e-6
BROWSER_SLOW_S = 3                   # a school PC takes perhaps 2-3× longer


def trains_in_browser(cfg, n_train, n_val):
    """True unless the browser would take over BROWSER_SLOW_S and over twice the server's time."""
    sizes = [cfg['n_inputs'], *(cfg['hidden_layers'] if cfg['model'] == 'neural_net' else []), 1]
    multiplies = sum(a * b for a, b in zip(sizes, sizes[1:]))      # one prediction of one point
    weights = multiplies + sum(sizes[1:])
    steps = cfg['epochs'] * math.ceil(n_train / min(batch_size(cfg, n_train), n_train))
    browser = (BROWSER_S_PER_MULTIPLY * cfg['epochs'] * multiplies * (6 * n_train + n_val)   # train ×3, evaluate, gradient ×2
               + BROWSER_S_PER_UPDATE * steps * weights)
    server = (steps + cfg['epochs']) * (SERVER_S_PER_STEP + SERVER_S_PER_LAYER_STEP * (len(sizes) - 1))   # + the gradient
    return browser <= BROWSER_SLOW_S or browser <= 2 * server


def public_options():
    """What the browser needs to build the settings form."""
    return {
        'defaults': DEFAULTS,
        'choices': {k: list(v) for k, v in CHOICES.items()},
        'ranges': RANGES,
        'presets': {'1': {k: v['expr'] for k, v in PRESETS_1D.items()},
                    '2': {k: v['expr'] for k, v in PRESETS_2D.items()}},
        'max_hidden_layers': MAX_HIDDEN_LAYERS,
        'max_neurons': MAX_NEURONS,
        'max_csv_bytes': MAX_CSV_BYTES,
    }
