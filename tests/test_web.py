import json

from conftest import login


def test_login_requires_roster_and_sets_password_on_first_login(client):
    r = login(client, '99999', '없는사람')
    assert r.status_code == 400 and r.get_json()['message'] == '학번과 이름이 명단에 없습니다.'
    r = login(client, confirm=False)
    assert r.status_code == 400 and r.get_json()['first_login']                       # asks to confirm
    assert login(client).get_json() == {'user': 1, 'home': None}                      # registers
    client.post('/logout')
    assert login(client, password='wrong', confirm=False).status_code == 400          # wrong password
    assert login(client, confirm=False).status_code == 200
    assert 'data-guest=""' in client.get('/').get_data(as_text=True)


def test_there_is_no_login_page(client):
    """/login (an old bookmark), a page that needs a login and a form post all open the playground's popup."""
    r = client.get('/login')
    assert r.status_code == 302 and r.headers['Location'] == '/?login=1'
    assert client.get('/login?next=/teacher/').headers['Location'] == '/?login=1&next=/teacher/'
    r = client.post('/login', data={'login_id': '20101', 'name': '김하늘', 'password': 'pass1234'})
    assert r.status_code == 302 and r.headers['Location'] == '/?login=1'
    assert client.post('/guest').status_code == 404
    r = client.get('/teacher/')
    assert r.status_code == 302 and r.headers['Location'] == '/?login=1&next=/teacher/'
    assert 'id="login-dialog"' in client.get(r.headers['Location']).get_data(as_text=True)


def test_login_popup_explains_the_password(client):
    html = client.get('/').get_data(as_text=True)
    assert '명단에 있는' not in html and 'class="info-btn"' in html
    assert '로그인은 등록된 사용자만 가능합니다.' in html
    assert '등록된 사용자의 경우 비밀번호는 첫 로그인 시 입력한 값으로 설정됩니다.' in html
    assert '트리 모델과 큰 신경망 모델' in html


def test_name_spaces_are_ignored(client):
    assert login(client, name='김 하늘').status_code == 200


def test_api_requires_login(client):
    assert client.get('/api/runs').status_code == 401
    assert client.post('/api/train', json={}).status_code == 401


def test_the_front_page_is_the_guest_playground_and_nothing_is_saved(client):
    page = client.get('/').get_data(as_text=True)                     # no login first: a guest
    assert 'data-guest="1"' in page and 'data-login="guest"' in page and 'id="login-dialog"' in page
    assert client.post('/api/preview', json={'epochs': 20}).status_code == 200
    prep = client.post('/api/prepare', json={'epochs': 20, 'model': 'linear'})
    assert prep.status_code == 200 and prep.get_json()['browser'] is True
    for method, path in [('post', '/api/train'), ('post', '/api/runs'), ('get', '/api/runs'), ('get', '/api/runs/1')]:
        r = getattr(client, method)(path, json={})
        assert r.status_code == 403 and r.get_json()['errors'] == ['게스트로 이용할 수 없는 기능입니다.'], path
    assert client.get('/teacher/').status_code == 302


def test_logging_out_leaves_the_guest_playground(client):
    client.get('/lang/en')
    login(client)
    page = client.get('/').get_data(as_text=True)
    assert 'data-guest=""' in page and 'data-login="20101"' in page
    r = client.post('/logout')
    assert r.status_code == 302 and r.headers['Location'] == '/'
    page = client.get('/').get_data(as_text=True)
    assert 'data-guest="1"' in page and '<html lang="en">' in page     # the language stays
    assert client.get('/api/runs').status_code == 403                # a guest now, not a lost session


def test_a_guest_logs_in_from_the_popup_and_keeps_the_runs(client):
    """The guest's login popup posts JSON to /login; the page then uploads the guest's runs (POST /api/runs)."""
    page = client.get('/').get_data(as_text=True)
    assert 'id="login-dialog"' in page and page.count('data-role="login-open"') == 2
    assert page.count('id="pw-help"') == 1 and 'data-role="pw-help-open"' in page and 'name="password_confirm"' in page
    who = {'login_id': '20101', 'name': '김 하늘', 'password': 'pass1234'}
    r = client.post('/login', json={**who, 'name': '없는사람'})
    assert r.status_code == 400 and r.get_json()['kind'] == 'error' and not r.get_json()['first_login']
    r = client.post('/login', json=who)                                  # first login: asks to confirm
    assert r.status_code == 400 and r.get_json()['first_login'] and r.get_json()['kind'] == 'info'
    assert client.post('/login', json={**who, 'password_confirm': 'other123'}).status_code == 400
    assert 'data-guest="1"' in client.get('/').get_data(as_text=True)    # still a guest
    r = client.post('/login', json={**who, 'password_confirm': 'pass1234'})
    assert r.status_code == 200 and r.get_json() == {'user': 1, 'home': None}
    end = {'status': 'done', 'steps': 20, 'final_train': 0.1, 'final_val': 0.2, 'duration': 0.3}
    assert client.post('/api/runs', json={'config': {'epochs': 20}, 'end': end, 'trainer': 1}).status_code == 201
    assert [row['seq'] for row in client.get('/api/runs').get_json()['runs']] == [1]
    assert 'data-guest=""' in client.get('/').get_data(as_text=True)
    client.post('/logout')
    assert client.post('/login', json={**who, 'password': 'wrong'}).get_json()['first_login'] is False
    assert client.post('/login', json=who).status_code == 200            # the password is set now


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
    assert login(client, 'teacher', '선생님', 'teachpw', confirm=False).get_json()['home'] == '/teacher/'
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
    students = client.get('/teacher/students.csv').get_data(as_text=True)
    assert students.lstrip('\ufeff').splitlines() == ['학번,이름,실험 수', '20101,김하늘,1', '20102,이바다,0']
    roster = '학번,이름\n20103,박산\n20101,김하늘\n'.encode('cp949')
    from io import BytesIO
    r = client.post('/teacher/roster', data={'roster': (BytesIO(roster), 'r.csv')}, follow_redirects=True)
    assert '박산' in r.get_data(as_text=True)
    client.post('/teacher/students/1/reset-password')
    client.post('/logout')
    assert login(client, confirm=False).get_json()['first_login']      # must register a new password


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
    page = client.get('/').get_data(as_text=True)
    assert 'icons/icon.png' in page and 'icons/icon.svg' not in page
    for path in ['/favicon.ico', '/apple-touch-icon.png']:
        r = client.get(path)
        assert r.status_code == 200 and r.content_type == 'image/png', path


def test_the_first_teacher_comes_from_admin_settings(tmp_path):
    from app import create_app
    from app.extensions import db
    from app.models import User
    cfg = {'TESTING': True, 'SQLALCHEMY_DATABASE_URI': f'sqlite:///{tmp_path}/a.db', 'SECRET_KEY': 'test',
           'ADMIN_LOGIN_ID': 'boss', 'ADMIN_NAME': '관리자', 'ADMIN_PASSWORD': 'first-pw'}
    app = create_app(cfg)
    client = app.test_client()
    r = client.post('/login', json={'login_id': 'boss', 'name': '관리자', 'password': 'first-pw'})
    assert r.get_json() == {'user': 1, 'home': '/teacher/'}
    app = create_app({**cfg, 'ADMIN_PASSWORD': 'other-pw'})      # a teacher exists: nothing changes
    with app.app_context():
        assert db.session.get(User, 1).check_password('first-pw')
        assert db.session.scalar(db.select(db.func.count()).select_from(User)) == 1
    app = create_app({**cfg, 'SQLALCHEMY_DATABASE_URI': f'sqlite:///{tmp_path}/b.db', 'ADMIN_PASSWORD': ''})
    with app.app_context():                                         # incomplete settings: no account
        assert db.session.scalar(db.select(db.func.count()).select_from(User)) == 0


def test_credits_come_from_settings(client, app):
    assert 'class="credits"' not in client.get('/').get_data(as_text=True)
    app.config.update(CREDIT_NAME='2026 Example School', CREDIT_URL='https://example.com',
                      GITHUB_URL='https://github.com/example', CONTACT_EMAIL='a@b.c')
    page = client.get('/').get_data(as_text=True)
    assert '<a href="https://example.com" target="_blank" rel="noopener">© 2026 Example School</a>' in page
    assert 'href="https://github.com/example"' in page
    assert 'a@b.c' not in page and '&#97;&#64;&#98;&#46;&#99;' in page


def test_the_first_teacher_adds_and_deletes_teachers(client):
    login(client, 'teacher', '선생님', 'teachpw', confirm=False)
    page = client.get('/teacher/').get_data(as_text=True)
    assert '최초 관리자' in page and 'action="/teacher/teachers"' in page
    r = client.post('/teacher/teachers', data={'login_id': 'second', 'name': '둘째'}, follow_redirects=True)
    assert '관리자 둘째을(를) 추가했습니다.' in r.get_data(as_text=True)
    r = client.post('/teacher/teachers', data={'login_id': '20101', 'name': '김하늘'}, follow_redirects=True)
    assert '아이디 20101은(는) 이미 사용 중입니다.' in r.get_data(as_text=True)     # a student stays a student
    client.post('/logout')
    r = login(client, 'second', '둘째', 'pw2nd', confirm=False)                     # no password yet: set it now
    assert r.status_code == 400 and r.get_json()['first_login']
    assert login(client, 'second', '둘째', 'pw2nd').get_json()['home'] == '/teacher/'
    page = client.get('/teacher/').get_data(as_text=True)                        # another teacher: no accounts part
    assert '최초 관리자' not in page and 'action="/teacher/teachers"' not in page
    assert client.post('/teacher/teachers', data={'login_id': 'third', 'name': '셋째'}).status_code == 403
    assert client.post('/teacher/teachers/1/delete').status_code == 403
    client.post('/logout')
    login(client, 'teacher', '선생님', 'teachpw', confirm=False)
    teacher_id, second_id = 3, 4                                                 # after the two students
    assert client.post(f'/teacher/teachers/{teacher_id}/delete').status_code == 404   # not the first teacher itself
    assert client.post('/teacher/teachers/1/delete').status_code == 404              # nor a student
    r = client.post(f'/teacher/teachers/{second_id}/delete', follow_redirects=True)
    assert '둘째' in r.get_data(as_text=True) and login(client, 'second', '둘째', 'pw2nd').status_code == 400
