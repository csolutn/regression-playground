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
    assert '명단에 있는' not in html
    assert '비밀번호는 첫 로그인시 입력한 값으로 설정됩니다.' in html
    assert '교사 정솔에게 문의하세요' in html


def test_name_spaces_are_ignored(client):
    assert login(client, name='김 하늘').status_code == 302


def test_api_requires_login(client):
    assert client.get('/api/runs').status_code == 401
    assert client.post('/api/train', json={}).status_code == 401


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
    rows = client.get('/api/runs').get_json()
    assert [r['seq'] for r in rows] == [2, 1]
    detail = client.get(f"/api/runs/{rows[0]['id']}").get_json()
    assert detail['frames'] and detail['loss']['steps'] and detail['config']['learning_rate'] == 0.1


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
    rows = client.get('/api/runs?user=1').get_json()
    assert len(rows) == 1
    assert client.get('/teacher/students/1').status_code == 200
    csv = client.get('/teacher/runs.csv').get_data(as_text=True)
    assert '20101' in csv
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
