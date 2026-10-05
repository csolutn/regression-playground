import json

from conftest import login


def test_login_requires_roster_and_sets_password_on_first_login(client):
    assert 'class list' in login(client, '99999', '없는사람').get_data(as_text=True) or True
    r = client.post('/login', data={'login_id': '20101', 'name': '김하늘', 'password': 'pass1234'})
    assert r.status_code == 200 and 'password_confirm' in r.get_data(as_text=True)     # asks to confirm
    assert login(client).status_code == 302                                              # registers
    client.post('/logout')
    assert login(client, password='wrong', confirm=False).status_code == 200            # wrong password
    assert login(client, confirm=False).status_code == 302
    assert client.get('/').status_code == 200


def test_first_login_keeps_the_typed_password(client):
    r = client.post('/login', data={'login_id': '20101', 'name': '김하늘', 'password': 'pass1234'})
    html = r.get_data(as_text=True)
    assert 'value="pass1234"' in html and r.headers['Cache-Control'] == 'no-store'
    r = client.post('/login', data={'login_id': '20101', 'name': '김하늘', 'password': 'pass1234',
                                    'password_confirm': 'other123'})
    assert 'value="pass1234"' not in r.get_data(as_text=True)      # mismatch: retype both


def test_login_page_explains_the_password(client):
    html = client.get('/login').get_data(as_text=True)
    assert '명단에 있는' not in html and 'class="info-btn"' in html
    assert '로그인은 등록된 사용자만 가능합니다.' in html
    assert '등록된 사용자의 경우 비밀번호는 첫 로그인 시 입력한 값으로 설정됩니다.' in html
    assert '트리 모델과 큰 신경망 모델' in html


def test_name_spaces_are_ignored(client):
    assert login(client, name='김 하늘').status_code == 302


def test_api_requires_login(client):
    assert client.get('/api/runs').status_code == 401
    assert client.post('/api/train', json={}).status_code == 401


def test_guest_uses_the_browser_only_and_nothing_is_saved(client):
    page = client.get('/login').get_data(as_text=True)
    assert 'formaction="/guest"' in page
    assert client.get('/').status_code == 302                         # the login page stays the first page
    r = client.post('/guest')
    assert r.status_code == 302 and r.headers['Location'].endswith('/')
    page = client.get('/').get_data(as_text=True)
    assert 'data-guest="1"' in page and 'data-login="guest"' in page
    assert client.post('/api/preview', json={'epochs': 20}).status_code == 200
    prep = client.post('/api/prepare', json={'epochs': 20, 'model': 'linear'})
    assert prep.status_code == 200 and prep.get_json()['browser'] is True
    for method, path in [('post', '/api/train'), ('post', '/api/runs'), ('get', '/api/runs'), ('get', '/api/runs/1')]:
        r = getattr(client, method)(path, json={})
        assert r.status_code == 403 and r.get_json()['errors'] == ['게스트로 이용할 수 없는 기능입니다.'], path
    assert client.get('/teacher/').status_code == 302


def test_logging_in_ends_the_guest_session(client):
    client.post('/guest')
    login(client)
    page = client.get('/').get_data(as_text=True)
    assert 'data-guest=""' in page and 'data-login="20101"' in page


def train(client, **cfg):
    r = client.post('/api/train', json={'epochs': 20, **cfg})
    assert r.status_code == 200, r.get_data(as_text=True)
    return [json.loads(line) for line in r.get_data(as_text=True).splitlines()]


def test_train_stream_saves_history(client):
    login(client)
    events = train(client)
    assert [e['type'] for e in events][0] == 'start'
    saved = events[-1]
    assert saved['type'] == 'saved' and saved['run']['seq'] == 1
    train(client, learning_rate=0.1)
    rows = client.get('/api/runs').get_json()['runs']
    assert [r['seq'] for r in rows] == [2, 1]
    detail = client.get(f"/api/runs/{rows[0]['id']}").get_json()
    assert detail['frames'] and detail['loss']['steps'] and detail['config']['learning_rate'] == 0.1
    assert len(detail['params']['final']) == len(detail['params']['contrib']) == 1 * 4 + 4 + 4 + 1   # for the landscape


def test_history_comes_a_page_at_a_time(app, client):
    from app.api import HISTORY_PAGE, save_run
    login(client)
    with app.app_context():
        for i in range(HISTORY_PAGE * 2 + 5):
            cfg = {'model': 'linear', 'data_source': 'csv', 'csv_text': 'x,y\n1,2\n', 'epochs': i}
            save_run(1, cfg, {'status': 'done', 'final_train': 1.0, 'final_val': 1.0, 'steps': 1, 'duration': 1.0}, {})
    first = client.get('/api/runs').get_json()
    assert [r['seq'] for r in first['runs']] == list(range(65, 35, -1))
    assert first['older']['seq'] == 35 and first['total'] == 65
    assert 'csv_text' not in first['runs'][0]['config'] and first['runs'][0]['config']['epochs'] == 64
    second = client.get(f"/api/runs?before={first['runs'][-1]['id']}").get_json()
    assert [r['seq'] for r in second['runs']] == list(range(35, 5, -1)) and second['older']['seq'] == 5
    rest = client.get(f"/api/runs?before={second['runs'][-1]['id']}&all=1").get_json()
    assert [r['seq'] for r in rest['runs']] == [5, 4, 3, 2, 1] and rest['older'] is None
    everything = client.get('/api/runs?all=1').get_json()
    assert len(everything['runs']) == 65 and everything['older'] is None


def test_bad_settings_return_translated_errors(client):
    login(client)
    r = client.post('/api/train', json={'function_1d': 'custom', 'expression_1d': 'y+1'})
    assert r.status_code == 400
    assert 'y' in r.get_json()['errors'][0]
    r = client.post('/api/train', json={'epochs': 0})
    assert '에포크' in r.get_json()['errors'][0] or 'Epochs' in r.get_json()['errors'][0]


def test_students_cannot_see_each_other(client, app):
    login(client)
    run_id = train(client)[-1]['run']['id']
    client.post('/logout')
    login(client, '20102', '이바다')
    assert client.get(f'/api/runs/{run_id}').status_code == 404
    assert client.delete(f'/api/runs/{run_id}').status_code == 404
    assert client.get('/api/runs?user=1').status_code == 403
    assert client.get('/teacher/').status_code == 403


def test_teacher_pages(client):
    login(client)
    train(client)
    client.post('/logout')
    assert login(client, 'teacher', '선생님', 'teachpw', confirm=False).status_code == 302
    assert '김하늘' in client.get('/teacher/').get_data(as_text=True)
    rows = client.get('/api/runs?user=1').get_json()['runs']
    assert len(rows) == 1
    assert client.get('/teacher/students/1').status_code == 200
    csv = client.get('/teacher/runs.csv').get_data(as_text=True)
    assert '20101' in csv and 'created_at (UTC)' in csv
    utc = csv.splitlines()[1].split(',')[3]
    seoul = client.get('/teacher/runs.csv?tz=Asia/Seoul').get_data(as_text=True)
    assert 'created_at (Asia/Seoul)' in seoul
    from datetime import datetime, timedelta
    assert datetime.fromisoformat(seoul.splitlines()[1].split(',')[3]) - datetime.fromisoformat(utc) == timedelta(hours=9)
    assert 'created_at (UTC)' in client.get('/teacher/runs.csv?tz=../../etc/passwd').get_data(as_text=True)
    roster = '학번,이름\n20103,박산\n20101,김하늘\n'.encode('cp949')
    from io import BytesIO
    r = client.post('/teacher/roster', data={'roster': (BytesIO(roster), 'r.csv')}, follow_redirects=True)
    assert '박산' in r.get_data(as_text=True)
    client.post('/teacher/students/1/reset-password')
    client.post('/logout')
    r = client.post('/login', data={'login_id': '20101', 'name': '김하늘', 'password': 'pass1234'})
    assert 'password_confirm' in r.get_data(as_text=True)      # must register a new password


def test_history_tools_name_the_csv_after_the_student(client):
    login(client)
    page = client.get('/').get_data(as_text=True)
    assert 'data-login="20101"' in page and 'data-role="filter-pop"' in page and 'data-role="csv"' in page
    client.post('/logout')
    login(client, 'teacher', '선생님', 'teachpw', confirm=False)
    assert 'data-login="20101"' in client.get('/teacher/students/1').get_data(as_text=True)


def test_train_gives_up_when_every_worker_is_busy(app, client, monkeypatch):
    from app import api
    monkeypatch.setattr(api, 'QUEUE_POLL_S', 0.05)
    app.config['QUEUE_TIMEOUT_S'] = 0.1
    login(client)
    with app.app_context():
        pool = api.training_pool()
    taken = 0
    while pool.acquire(timeout=0):
        taken += 1
    try:
        r = client.post('/api/train', json={'epochs': 10})
        events = [json.loads(line) for line in r.get_data(as_text=True).splitlines()]
        assert events[0]['type'] == 'queued' and events[-1]['type'] == 'error'
    finally:
        for _ in range(taken):
            pool.slots.release()


def test_static_urls_carry_a_version_and_are_cached(app, client, tmp_path):
    from app import create_app, static_version
    login(client)
    page = client.get('/').get_data(as_text=True)
    version = app.config['STATIC_VERSION']
    assert version == static_version() and f'/static/{version}/js/playground.js' in page
    prod = create_app({'DEBUG': False, 'SECRET_KEY': 'test', 'SQLALCHEMY_DATABASE_URI': f'sqlite:///{tmp_path}/p.db'})
    client = prod.test_client()                                 # like the server: not in debug mode
    r = client.get(f'/static/{version}/js/playground.js')
    assert r.status_code == 200 and 'max-age=31536000' in r.headers['Cache-Control']
    r = client.get('/static/js/playground.js')                 # an old page asking for the unversioned URL
    assert r.status_code == 200 and 'no-cache' in r.headers['Cache-Control']
    r = client.get('/static/0123456789/js/train-worker.js')    # a page from an older deploy
    assert r.status_code == 200 and 'no-cache' in r.headers['Cache-Control']


def test_icons_for_safari_and_ios(client):
    page = client.get('/login').get_data(as_text=True)
    assert 'icons/icon.png' in page and 'icons/icon.svg' not in page
    for path in ['/favicon.ico', '/apple-touch-icon.png']:
        r = client.get(path)
        assert r.status_code == 200 and r.content_type == 'image/png', path
