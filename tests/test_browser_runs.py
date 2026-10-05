"""Linear and neural network runs that train in the browser: /api/prepare and POST /api/runs."""
import json

from conftest import login

from app.ml import data as ml_data
from app.ml import options

HEAVY_BGD = {'hidden_layers': [32] * 6, 'data_size': 2000, 'epochs': 1500, 'optimizer': 'bgd'}
HEAVY_SGD = {'hidden_layers': [32] * 3, 'data_size': 500, 'epochs': 1000, 'optimizer': 'sgd'}


def post(client, url, body):
    return client.post(url, data=json.dumps(body), content_type='application/json')


def upload_for(over):
    """What the browser uploads: the settings, the end of the run, and its trainer's version."""
    cfg, errors = options.validate(over)
    assert not errors
    end = {'type': 'end', 'status': 'done', 'steps': cfg['epochs'], 'final_train': 0.0123, 'final_val': 0.0234, 'duration': 0.4}
    return {'config': cfg, 'end': end, 'trainer': 1}


def test_slow_big_batch_runs_stay_on_the_server():
    def in_browser(over):
        cfg, _ = options.validate(over)
        n_val = int(cfg['data_size'] * cfg['validation_ratio'])
        return options.trains_in_browser(cfg, cfg['data_size'] - n_val, n_val)
    assert in_browser({})                           # the default run
    assert in_browser(HEAVY_SGD)                    # many small steps: slow in torch, fine in JavaScript
    assert not in_browser(HEAVY_BGD)                # big matrices: under a second in torch


def test_prepare(client):
    login(client)
    r = post(client, '/api/prepare', {})
    body = r.get_json()
    assert body['browser'] and body['time_limit_s'] > 0 and body['meta']['structure'] == [1, 4, 1]
    assert post(client, '/api/prepare', HEAVY_BGD).get_json() == {'browser': False}
    assert post(client, '/api/prepare', {'model': 'random_forest'}).status_code == 400
    assert post(client, '/api/prepare', {'epochs': 0}).status_code == 400          # invalid settings


def test_upload_saves_only_the_results(client):
    """The animation is not stored: the browser trains the run again from its settings to show it."""
    login(client)
    upload = upload_for({'n_inputs': 2, 'epochs': 20, 'seed': 7})
    r = post(client, '/api/runs', upload)
    assert r.status_code == 201 and r.get_json()['run']['seq'] == 1
    detail = client.get(f"/api/runs/{r.get_json()['run']['id']}").get_json()
    assert detail['run']['final_val'] == 0.0234 and detail['run']['steps'] == 20
    assert detail['config']['seed'] == 7 and detail['config']['n_inputs'] == 2      # all it takes to train it again
    assert detail['replay'] == {'trainer': 1, 'data': ml_data.DATA_VERSION, 'data_changed': False}
    assert 'frames' not in detail and 'params' not in detail


def test_replay_notices_changed_data(app, client, monkeypatch):
    login(client)
    run_id = post(client, '/api/runs', upload_for({'epochs': 20})).get_json()['run']['id']
    monkeypatch.setattr(ml_data, 'DATA_VERSION', ml_data.DATA_VERSION + 1)
    assert client.get(f'/api/runs/{run_id}').get_json()['replay']['data_changed'] is True


def test_upload_rejects_bad_runs(client):
    login(client)
    good = upload_for({'epochs': 20})
    assert post(client, '/api/runs', good).status_code == 201

    def bad(change):
        body = json.loads(json.dumps(good))
        change(body)
        return post(client, '/api/runs', body).status_code
    assert bad(lambda b: b['end'].__setitem__('status', 'great')) == 400
    assert bad(lambda b: b['end'].__setitem__('steps', 21)) == 400               # more epochs than set
    assert bad(lambda b: b['end'].__setitem__('final_val', float('nan'))) == 400  # json.dumps writes NaN
    assert bad(lambda b: b['end'].pop('duration')) == 400
    assert bad(lambda b: b['config'].__setitem__('model', 'decision_tree')) == 400
    assert bad(lambda b: b.pop('end')) == 400
    assert bad(lambda b: b.pop('trainer')) == 400
    assert bad(lambda b: b.__setitem__('trainer', '1')) == 400
    assert client.post('/api/runs', data='x', content_type='text/plain').status_code == 415


def test_upload_requires_login(client):
    assert post(client, '/api/runs', {}).status_code == 401
    assert post(client, '/api/prepare', {}).status_code == 401
