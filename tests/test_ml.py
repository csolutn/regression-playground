import json
from pathlib import Path

import pytest

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
