"""② prediction → ③ loss → ④ optimization, streamed as events.

``train(cfg)`` is a generator of plain dicts, so it can be tested without Flask:

  {'type': 'start', 'meta': {...}}                        data points, plot inputs, true function
  {'type': 'frame', 'step', 'pred', 'train', 'val'}       one animation frame (prediction on the plot inputs)
  {'type': 'loss', 'steps': [...], 'train': [...], 'val': [...]}   loss history since the last event
  {'type': 'params', 'init': [...], 'final': [...], 'contrib': [...], 'path_epochs': [...],
   'paths': [{'index', 'values', 'lowered'}, ...]}
        linear / neural network, not diverged: the weights (flat: W0, b0, W1, b1, …; W is out × in) at the
        start and at the end, how much each one lowered the training loss, and for the TOP_WEIGHTS that
        lowered it most their value and how much they had lowered it at every frame and, early on, more
        often (at path_epochs, which may be parts of an epoch)
  {'type': 'end', 'final_train', 'final_val', 'steps', 'duration', 'status'}

Losses are measured on standardized y (as in the notebook), so different data and models compare fairly.
"""
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from .data import build_dataset, expression_for
from .options import GRADIENT_MODELS, PLAIN_GD, batch_size, step_budget_error

torch.set_num_threads(1)   # many small jobs in parallel, not one big one
torch.backends.mkldnn.enabled = False   # oneDNN (Linux builds) makes these tiny ops ~6× slower

ACTIVATIONS = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU, 'gelu': nn.GELU,
               'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'linear': nn.Identity}
MAX_FRAMES = 300           # one frame per epoch (or tree) up to this many, then every k-th
CHECKPOINT_GROWTH = 1.2    # weight contributions: checkpoints after steps 1, 2, 3, 4, 5, 6, 8, 10, …, and every epoch
MAX_SPLITS = 4             # … with a stretch between two of them halved at most this many times
TOP_WEIGHTS = 4            # the weights with the largest contributions: their paths (values, contributions so far) are saved
LOSS_EVENTS = 100          # about this many loss events per run
TIME_LIMIT_S = 30          # stop a run that takes longer than this (saved as 'timeout')


class BudgetError(ValueError):
    def __init__(self, err):
        super().__init__(err[0])
        self.msgid, self.params = err


def structure(cfg):
    hidden = cfg['hidden_layers'] if cfg['model'] == 'neural_net' else []
    return [cfg['n_inputs'], *hidden, 1]


def n_params(cfg):
    """How many weights and biases the network has."""
    sizes = structure(cfg)
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def build_network(cfg):
    sizes, layers = structure(cfg), []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(n_in, n_out))
        if i < len(sizes) - 2:          # no activation on the output layer
            layers.append(ACTIVATIONS[cfg['activation']]())
    return nn.Sequential(*layers)


def build_optimizer(cfg, params):
    lr = cfg['learning_rate']
    return {
        **dict.fromkeys(PLAIN_GD, lambda: torch.optim.SGD(params, lr=lr)),
        'momentum': lambda: torch.optim.SGD(params, lr=lr, momentum=0.9),
        'rmsprop': lambda: torch.optim.RMSprop(params, lr=lr),
        'adam': lambda: torch.optim.Adam(params, lr=lr),
    }[cfg['optimizer']]()


def build_loss(cfg):
    return {'mse': nn.MSELoss, 'mae': nn.L1Loss,
            'huber': lambda: nn.HuberLoss(delta=cfg['huber_delta'])}[cfg['loss']]()


def loss_np(cfg, pred, true):
    """Same definitions as build_loss, for the tree models."""
    e = pred - true
    if cfg['loss'] == 'mse':
        return float(np.mean(e ** 2))
    if cfg['loss'] == 'mae':
        return float(np.mean(np.abs(e)))
    d, a = cfg['huber_delta'], np.abs(e)
    return float(np.mean(np.where(a <= d, 0.5 * e ** 2, d * (a - 0.5 * d))))


def rounded(a, digits=4):
    """Shorter JSON: keep ``digits`` significant digits."""
    return [float(f'{v:.{digits}g}') for v in np.asarray(a, dtype=float).ravel()]


def check(cfg):
    """Build the data and check the step budget; raises DataError or BudgetError."""
    ds = build_dataset(cfg)
    if cfg['model'] in GRADIENT_MODELS and (err := step_budget_error(cfg, len(ds.X_train))):
        raise BudgetError(err)
    return ds


def train(cfg, time_limit=TIME_LIMIT_S):
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    ds = check(cfg)

    yield {'type': 'start', 'meta': meta(cfg, ds)}
    started = time.monotonic()
    history = {'steps': [], 'train': [], 'val': []}
    status = 'done'
    run = train_gradient if cfg['model'] in GRADIENT_MODELS else train_trees
    for event in run(cfg, ds, started + time_limit):
        if event['type'] == 'loss':
            for k in history:
                history[k] += event[k]
        if event['type'] == 'status':      # 'timeout' or 'diverged'
            status = event['status']
            continue
        yield event
    yield {'type': 'end', 'status': status, 'steps': history['steps'][-1],
           'final_train': history['train'][-1], 'final_val': history['val'][-1],
           'duration': round(time.monotonic() - started, 2)}


def meta(cfg, ds):
    return {
        'n_inputs': ds.n_inputs,
        'feature_names': ds.feature_names,
        'target_name': ds.target_name,
        'x_train': [rounded(c) for c in ds.X_train.T], 'y_train': rounded(ds.y_train),
        'x_val': [rounded(c) for c in ds.X_val.T], 'y_val': rounded(ds.y_val),
        'axes': [rounded(a) for a in ds.axes],
        'y_true': rounded(ds.y_plot_true) if ds.y_plot_true is not None else None,
        'expression': expression_for(cfg) if cfg['data_source'] == 'function' else None,
        'structure': structure(cfg),
        'model': cfg['model'],
        'loss': cfg['loss'],
        'total_steps': total_steps(cfg),
    }


def total_steps(cfg):
    if cfg['model'] in GRADIENT_MODELS:
        return cfg['epochs']
    return cfg['max_depth'] if cfg['model'] == 'decision_tree' else cfg['n_estimators']


def frame(ds, step, pred_scaled, train_loss, val_loss):
    finite = lambda v: float(f'{v:.6g}') if math.isfinite(v) else None
    return {'type': 'frame', 'step': step, 'pred': rounded(ds.unscale_y(pred_scaled)),
            'train': finite(train_loss), 'val': finite(val_loss)}


def to_tensor(a):
    return torch.tensor(a, dtype=torch.float32).reshape(len(a), -1)


def train_gradient(cfg, ds, deadline):
    model = build_network(cfg)
    optimizer, loss_fn = build_optimizer(cfg, model.parameters()), build_loss(cfg)
    Xt, yt = to_tensor(ds.scale_x(ds.X_train)), to_tensor(ds.scale_y(ds.y_train))
    Xv, yv = to_tensor(ds.scale_x(ds.X_val)), to_tensor(ds.scale_y(ds.y_val))
    Xp = to_tensor(ds.scale_x(ds.X_plot))
    bs = min(batch_size(cfg, len(Xt)), len(Xt))
    gen = torch.Generator().manual_seed(cfg['seed'])
    epochs = cfg['epochs']
    frame_every = max(1, math.ceil(epochs / MAX_FRAMES))
    loss_every = max(1, math.ceil(epochs / LOSS_EVENTS))
    pending = {'steps': [], 'train': [], 'val': []}

    contributions = Contributions(model, loss_fn, Xt, yt)

    def evaluate():
        """Training loss (at a checkpoint of the contributions) and validation loss."""
        model.eval()
        tl = contributions.checkpoint()
        with torch.no_grad():
            return tl, loss_fn(model(Xv), yv).item()

    tl, vl = evaluate()                   # epoch 0: the untrained model
    contributions.snapshot(0, 0)
    with torch.no_grad():
        yield frame(ds, 0, model(Xp).numpy(), tl, vl)

    steps, next_checkpoint = 0, 1
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), bs):
            b = order[i:i + bs]
            pred = model(Xt[b])           # ② prediction
            loss = loss_fn(pred, yt[b])   # ③ loss
            optimizer.zero_grad()
            loss.backward()               #    gradient (backpropagation)
            optimizer.step()              # ④ optimization: update the weights
            steps += 1
            if steps >= next_checkpoint:  # the weights move fast early on: checkpoints in the epoch too
                next_checkpoint = math.ceil(steps * CHECKPOINT_GROWTH)
                if i + bs < len(Xt):      # (the epoch's end has its own)
                    contributions.checkpoint()
                    if steps >= contributions.next_snapshot:
                        contributions.snapshot(epoch - 1 + (i + len(b)) / len(Xt), steps)

        tl, vl = evaluate()
        diverged = not (math.isfinite(tl) and math.isfinite(vl))
        for k, v in (('steps', epoch), ('train', tl), ('val', vl)):
            pending[k].append(v if k == 'steps' else (float(f'{v:.6g}') if math.isfinite(v) else None))
        last = epoch == epochs or diverged or time.monotonic() > deadline
        is_frame = epoch % frame_every == 0 or last
        if not diverged and (is_frame or steps >= contributions.next_snapshot):
            contributions.snapshot(epoch, steps)
        if is_frame:
            with torch.no_grad():
                pred = np.nan_to_num(model(Xp).numpy(), nan=ds.y_mean)
            yield frame(ds, epoch, pred, tl, vl)
        if epoch % loss_every == 0 or last:
            yield {'type': 'loss', **pending}
            pending = {'steps': [], 'train': [], 'val': []}
        if last:
            if diverged:
                yield {'type': 'status', 'status': 'diverged'}
                return
            if epoch < epochs:
                yield {'type': 'status', 'status': 'timeout'}
            if params := contributions.result():
                yield {'type': 'params', **params}
            return


class Contributions:
    """How much each weight lowered the training loss (Loss Change Allocation): between two checkpoints,
    weight i moved by Δθᵢ while the full-batch gradient went from g to g', so it lowered the loss by
    about −½(gᵢ + g'ᵢ)·Δθᵢ. Summed over the weights this should be the drop of the training loss between
    them; where it is not (big steps across a curved valley), the stretch is halved and each half added
    up the same way, down to MAX_SPLITS times.
    """
    def __init__(self, model, loss_fn, Xt, yt):
        self.model, self.loss_fn, self.Xt, self.yt = model, loss_fn, Xt, yt
        self.params = list(model.parameters())
        self.total = torch.zeros(sum(p.numel() for p in self.params), dtype=torch.float64)
        self.init = self.prev = None
        self.snapshots, self.next_snapshot = [], 1

    def snapshot(self, epoch, steps):
        """Every weight and its contribution so far, for the paths of the TOP_WEIGHTS: at every frame and, where
        the weights move fast early on, at checkpoints after steps 1, 2, 3, 4, 5, 6, 8, 10, … too (epoch may be
        part of an epoch). Right after a checkpoint."""
        weights = torch.cat([p.detach().reshape(-1) for p in self.params]).clone()
        self.snapshots.append((epoch, weights, self.total.clone()))
        self.next_snapshot = max(steps + 1, math.ceil(steps * CHECKPOINT_GROWTH))

    def point(self):
        """(training loss, weights, full-batch gradient) at the current weights; weights flat as in the docstring."""
        self.model.zero_grad()
        loss = self.loss_fn(self.model(self.Xt), self.yt)
        loss.backward()
        flat = lambda ts: torch.cat([t.detach().reshape(-1) for t in ts]).double()
        return loss.item(), flat(self.params), flat(p.grad for p in self.params)

    def along(self, a, b, tol, depth=0):
        """Add the contributions along the straight line from point a to point b, if they add up to its
        drop within tol (else the halves', each within tol / 2); True if it was split."""
        (l0, p0, g0), (l1, p1, g1) = a, b
        part = -0.5 * (g0 + g1) * (p1 - p0)
        if depth == MAX_SPLITS or abs(part.sum().item() - (l0 - l1)) <= tol:
            self.total += part
            return False
        with torch.no_grad():
            nn.utils.vector_to_parameters((0.5 * (p0 + p1)).float(), self.params)
        half = self.point()
        self.along(a, half, tol / 2, depth + 1)
        self.along(half, b, tol / 2, depth + 1)
        return True

    def checkpoint(self):
        """Add up the contributions since the last checkpoint; returns the training loss."""
        cur = self.point()
        if self.prev is None:
            self.init = cur
        elif self.along(self.prev, cur, 0.02 * abs(self.prev[0] - cur[0]) + 1e-5 * self.init[0]):
            with torch.no_grad():                 # back from halfway
                nn.utils.vector_to_parameters(cur[1].float(), self.params)
        self.prev = cur
        return cur[0]

    def result(self):
        """The 'params' event without its type, or None if anything is not finite (a diverged run)."""
        final = self.prev[1]
        if not (torch.isfinite(final).all() and torch.isfinite(self.total).all()):
            return None
        contrib = rounded(self.total, 6)
        epochs, weights, lowered = zip(*self.snapshots)
        weights, lowered = torch.stack(weights), torch.stack(lowered)
        top = sorted(range(len(contrib)), key=lambda i: -contrib[i])[:TOP_WEIGHTS]
        return {'init': rounded(self.init[1], 6), 'final': rounded(final, 6), 'contrib': contrib,
                'path_epochs': rounded(epochs, 6), 'paths': [{'index': i, 'values': rounded(weights[:, i], 6), 'lowered': rounded(lowered[:, i], 6)}
                          for i in top]}


def train_trees(cfg, ds, deadline):   # sklearn fits are quick; the deadline is not checked
    Xt, Xv, Xp = ds.scale_x(ds.X_train), ds.scale_x(ds.X_val), ds.scale_x(ds.X_plot)
    yt, yv = ds.scale_y(ds.y_train), ds.scale_y(ds.y_val)
    criterion = 'absolute_error' if cfg['loss'] == 'mae' else 'squared_error'
    depth, seed = cfg['max_depth'], cfg['seed']

    if cfg['model'] == 'decision_tree':          # grow the depth 1 → max_depth
        def stages():
            for d in range(1, depth + 1):
                m = DecisionTreeRegressor(max_depth=d, criterion=criterion, random_state=seed).fit(Xt, yt)
                yield d, m.predict(Xt), m.predict(Xv), m.predict(Xp)
    elif cfg['model'] == 'random_forest':        # add trees one by one and average them
        rf = RandomForestRegressor(n_estimators=cfg['n_estimators'], max_depth=depth, criterion=criterion,
                                   random_state=seed).fit(Xt, yt)

        def stages():
            st = sv = sp = 0
            for k, tree in enumerate(rf.estimators_, 1):
                st, sv, sp = st + tree.predict(Xt), sv + tree.predict(Xv), sp + tree.predict(Xp)
                yield k, st / k, sv / k, sp / k
    else:                                        # boosting: each tree corrects the previous error
        gb_loss = {'mse': 'squared_error', 'mae': 'absolute_error', 'huber': 'huber'}[cfg['loss']]
        gb = GradientBoostingRegressor(n_estimators=cfg['n_estimators'], max_depth=depth, loss=gb_loss,
                                       learning_rate=cfg['tree_learning_rate'], random_state=seed).fit(Xt, yt)

        def stages():
            yield from zip(range(1, cfg['n_estimators'] + 1),
                           gb.staged_predict(Xt), gb.staged_predict(Xv), gb.staged_predict(Xp))

    n_steps = total_steps(cfg)
    every = max(1, math.ceil(n_steps / MAX_FRAMES))
    history = {'steps': [], 'train': [], 'val': []}
    for step, pt, pv, pp in stages():
        tl, vl = float(f'{loss_np(cfg, pt, yt):.6g}'), float(f'{loss_np(cfg, pv, yv):.6g}')
        for k, v in (('steps', step), ('train', tl), ('val', vl)):
            history[k].append(v)
        if step == 1 or step % every == 0 or step == n_steps:
            yield frame(ds, step, pp, tl, vl)
    yield {'type': 'loss', **history}
