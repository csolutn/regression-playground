"""Linear and neural network runs that train in the browser: /api/prepare and POST /api/runs."""
import json

from conftest import login

from app.ml import options, trainer

HEAVY_BGD = {'hidden_layers': [32] * 6, 'data_size': 2000, 'epochs': 1500, 'optimizer': 'bgd'}
HEAVY_SGD = {'hidden_layers': [32] * 3, 'data_size': 500, 'epochs': 1000, 'optimizer': 'sgd'}


def post(client, url, body):
    return client.post(url, data=json.dumps(body), content_type='application/json')


def upload_for(over):
    """What the browser uploads, made here with the server's own trainer (same shapes)."""
    cfg, errors = options.validate(over)
    assert not errors
    events = list(trainer.train(cfg))
    loss = {k: [v for e in events if e['type'] == 'loss' for v in e[k]] for k in ('steps', 'train', 'val')}
    frames = [{k: e[k] for k in ('step', 'pred', 'train', 'val')} for e in events if e['type'] == 'frame']
    end = {k: v for k, v in events[-1].items() if k != 'type'}
    return {'config': cfg, 'frames': frames, 'loss': loss, 'end': end}


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


def test_upload_saves_a_browser_run(client):
    login(client)
    upload = upload_for({'n_inputs': 2, 'epochs': 20})
    r = post(client, '/api/runs', upload)
    assert r.status_code == 201 and r.get_json()['run']['seq'] == 1
    detail = client.get(f"/api/runs/{r.get_json()['run']['id']}").get_json()
    assert detail['meta']['n_inputs'] == 2 and len(detail['frames']) == len(upload['frames'])
    assert detail['run']['final_val'] == upload['end']['final_val']


def test_upload_rejects_bad_runs(client):
    login(client)
    good = upload_for({'epochs': 20})
    assert post(client, '/api/runs', good).status_code == 201

    def bad(change):
        body = json.loads(json.dumps(good))
        change(body)
        return post(client, '/api/runs', body).status_code
    assert bad(lambda b: b['frames'][0]['pred'].pop()) == 400                     # wrong number of points
    assert bad(lambda b: b['frames'][0]['pred'].__setitem__(0, 'x')) == 400
    assert bad(lambda b: b['end'].__setitem__('status', 'great')) == 400
    assert bad(lambda b: b['loss']['val'].pop()) == 400
    assert bad(lambda b: b['end'].__setitem__('steps', 21)) == 400               # more epochs than set
    assert bad(lambda b: b['config'].__setitem__('model', 'decision_tree')) == 400
    assert bad(lambda b: b.pop('frames')) == 400
    assert bad(lambda b: b['frames'][0].__setitem__('val', float('nan'))) == 400  # json.dumps writes NaN
    assert client.post('/api/runs', data='x', content_type='text/plain').status_code == 415


def test_upload_requires_login(client):
    assert post(client, '/api/runs', {}).status_code == 401
    assert post(client, '/api/prepare', {}).status_code == 401
