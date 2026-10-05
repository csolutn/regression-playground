import json
from pathlib import Path

import pytest
import torch

from app.ml import data, expr, options, trainer


def run(**overrides):
    cfg, errors = options.validate(overrides)
    assert not errors, errors
    return list(trainer.train(cfg))


@pytest.mark.parametrize('model', ['linear', 'neural_net', 'decision_tree', 'random_forest', 'gradient_boosting'])
@pytest.mark.parametrize('n_inputs', [1, 2])
def test_every_model_streams_start_frames_end(model, n_inputs):
    events = run(model=model, n_inputs=n_inputs, epochs=20, n_estimators=10)
    assert events[0]['type'] == 'start' and events[-1]['type'] == 'end'
    frames = [e for e in events if e['type'] == 'frame']
    assert len(frames) >= 2
    grid = len(events[0]['meta']['axes'][0]) * (len(events[0]['meta']['axes'][1]) if n_inputs == 2 else 1)
    assert all(len(f['pred']) == grid for f in frames)
    json.dumps(events, allow_nan=False)      # must be valid JSON for the browser


@pytest.mark.parametrize('over', [
    {'model': 'neural_net', 'hidden_layers': [4, 4], 'optimizer': 'adam'},
    {'model': 'neural_net', 'hidden_layers': [4, 4], 'activation': 'tanh', 'optimizer': 'bgd', 'learning_rate': 0.3, 'epochs': 300},
    {'model': 'neural_net', 'hidden_layers': [8], 'optimizer': 'sgd', 'epochs': 30},
    {'model': 'linear', 'n_inputs': 2, 'epochs': 100},
])
def test_weights_and_their_contributions(over):
    """The saved weights give the run's losses, and the contributions add up to the drop of the training loss."""
    events = run(**over)
    cfg = options.validate(over)[0]
    params = next(e for e in events if e['type'] == 'params')
    frames = [e for e in events if e['type'] == 'frame']
    assert len(params['init']) == len(params['final']) == len(params['contrib']) == trainer.n_params(cfg)

    ds = data.build_dataset(cfg)
    model, loss_fn = trainer.build_network(cfg), trainer.build_loss(cfg)
    Xt, yt = trainer.to_tensor(ds.scale_x(ds.X_train)), trainer.to_tensor(ds.scale_y(ds.y_train))

    def loss_at(flat):
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(torch.tensor(flat, dtype=torch.float32), model.parameters())
            return loss_fn(model(Xt), yt).item()
    first, last = frames[0]['train'], frames[-1]['train']
    assert loss_at(params['init']) == pytest.approx(first, rel=1e-3)
    assert loss_at(params['final']) == pytest.approx(last, rel=1e-2, abs=1e-5)
    assert sum(params['contrib']) == pytest.approx(first - last, rel=0.05)

    # the paths of the TOP_WEIGHTS largest contributions: at every frame and more often early on
    top = sorted(range(len(params['contrib'])), key=lambda i: -params['contrib'][i])[:trainer.TOP_WEIGHTS]
    assert [p['index'] for p in params['paths']] == top
    times = params['path_epochs']
    assert times == sorted(times) and set(f['step'] for f in frames) <= set(times)
    for p in params['paths']:
        assert len(p['values']) == len(times)
        assert p['values'][0] == pytest.approx(params['init'][p['index']], abs=1e-5)
        assert p['values'][-1] == pytest.approx(params['final'][p['index']], abs=1e-5)
        assert len(p['lowered']) == len(times) and p['lowered'][0] == 0
        assert p['lowered'][-1] == pytest.approx(params['contrib'][p['index']], rel=1e-4)


def test_diverged_and_tree_runs_save_no_weights():
    events = run(model='neural_net', optimizer='sgd', learning_rate=1, hidden_layers=[32, 32, 32], epochs=50)
    assert events[-1]['status'] == 'diverged' and not any(e['type'] == 'params' for e in events)
    assert not any(e['type'] == 'params' for e in run(model='decision_tree'))


def test_neural_net_learns_abs():
    end = run(model='neural_net', hidden_layers=[8, 8], optimizer='adam', epochs=300)[-1]
    assert end['status'] == 'done' and end['final_val'] < 0.05


def test_divergence_is_reported_not_crashing():
    end = run(model='neural_net', optimizer='sgd', learning_rate=1, hidden_layers=[32, 32, 32], epochs=50)[-1]
    assert end['status'] in ('diverged', 'done')


def test_plain_gradient_descent_fixes_the_batch():
    for optimizer in ('bgd', 'mini', 'sgd'):
        cfg, errors = options.validate({'optimizer': optimizer, 'batch_method': 'mini', 'epochs': 300})
        assert not errors and cfg['batch_method'] == optimizer
    assert options.batch_size(options.validate({'optimizer': 'sgd', 'epochs': 300})[0], 400) == 1


def test_validation_rejects_bad_values():
    _, errors = options.validate({'model': 'svm', 'epochs': 0, 'x_min': 3, 'x_max': 1})
    assert len(errors) >= 3
    _, errors = options.validate({'optimizer': 'adam', 'batch_method': 'sgd', 'epochs': 1500, 'data_size': 2000})
    assert any('Too much computation' in e[0] for e in errors)
    _, errors = options.validate({'epochs': 3000})
    assert errors                                                  # above the maximum of 1500


def test_sgd_fits_the_budget_at_the_default_epochs():
    _, errors = options.validate({'optimizer': 'sgd'})            # default data and epochs
    assert not errors


@pytest.mark.parametrize('text', ['__import__("os")', 'x.__class__', '().__class__', 'open("f")', 'lambda: 1', '[x]'])
def test_formula_sandbox(text):
    import numpy as np
    with pytest.raises(expr.ExpressionError):
        expr.evaluate(text, {'x': np.zeros(3)})


def test_csv_data():
    csv = 'hours,sleep,score\n' + '\n'.join(f'{i},{i % 7},{2 * i + 1}' for i in range(40))
    events = run(data_source='csv', csv_text=csv, csv_features=['hours'], csv_target='score', epochs=10)
    assert events[0]['meta']['feature_names'] == ['hours']
    cfg, _ = options.validate({'data_source': 'csv', 'csv_text': csv, 'csv_features': ['nope'], 'csv_target': 'score'})
    with pytest.raises(data.DataError):
        data.build_dataset(cfg)


@pytest.mark.parametrize('path', sorted(Path(__file__).parent.parent.glob('app/static/samples/*.csv')), ids=lambda p: p.stem)
def test_sample_csvs_train(path):
    text = path.read_text()
    cols = text.splitlines()[0].split(',')
    events = run(data_source='csv', csv_text=text, csv_features=cols[:-1], csv_target=cols[-1], epochs=10)
    assert events[-1]['status'] == 'done'
