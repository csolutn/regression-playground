"""Teachers of other schools (a prefix), advanced features, and the data of the results view and HTML backup."""
import json
import shutil
import subprocess
from io import BytesIO

import pytest
from conftest import login
from test_browser_runs import HEAVY_BGD, post, upload_for

from app.admin import REPORT_MODULES
from app.bundle import bundle

ADMIN = ('teacher', '선생님', 'teachpw')


def as_admin(client):
    client.post('/logout')
    assert login(client, *ADMIN, confirm=False).status_code == 200


def add_other_school(client, prefix='abc'):
    """The system admin adds a teacher with a prefix, who logs in and imports a class list (numbers only)."""
    as_admin(client)
    r = client.post('/teacher/teachers', data={'login_id': 'kim', 'name': '김교사', 'prefix': prefix.upper(),
                                               'email': 'kim@example.org', 'memo': '메일로 신청'}, follow_redirects=True)
    assert '관리자 김교사을(를) 추가했습니다.' in r.get_data(as_text=True)
    client.post('/logout')
    assert login(client, 'kim', '김교사', 'kimpw').status_code == 200
    roster = 'student_id,name\n21013,홍길동\nabc21014,이둘\n'.encode()
    client.post('/teacher/roster', data={'roster': (BytesIO(roster), 'r.csv')})


def test_a_prefixed_teacher_sees_only_their_own_students(client):
    add_other_school(client)
    page = client.get('/teacher/').get_data(as_text=True)
    assert '홍길동' in page and '이둘' in page and '김하늘' not in page
    assert '<span class="id-prefix">abc</span>21013' in page and '시스템 관리자' not in page
    assert client.get('/teacher/students.csv').get_data(as_text=True).lstrip('﻿').splitlines()[1:] == \
        ['21013,홍길동,0', '21014,이둘,0']                                      # the numbers as typed, without the prefix
    assert client.get('/teacher/students/1').status_code == 404                # 김하늘 (no prefix) is not theirs
    assert client.get('/teacher/report.json?student=1').status_code == 404
    assert client.get('/api/runs?user=1').status_code == 403
    assert client.post('/teacher/students/1/delete').status_code == 404
    assert [s['number'] for s in client.get('/teacher/report.json').get_json()['students']] == ['21013', '21014']
    as_admin(client)                                                            # the system admin sees everyone
    page = client.get('/teacher/').get_data(as_text=True)
    assert '홍길동' in page and '김하늘' in page and '메일로 신청' in page and 'kim@example.org' in page


def test_students_of_another_school_log_in_with_the_prefix(client):
    add_other_school(client)
    client.post('/logout')
    assert login(client, '21013', '홍길동').status_code == 400                 # not the home school's 21013
    assert login(client, 'ABC21013', '홍길동').status_code == 200               # capitals are fine


def test_advanced_features_are_off_for_another_school_until_turned_on(client):
    add_other_school(client)
    client.post('/logout')
    login(client, 'abc21013', '홍길동')
    page = client.get('/').get_data(as_text=True)
    assert 'value="decision_tree"' not in page and '"advanced": false' in page
    assert post(client, '/api/prepare', HEAVY_BGD).get_json()['browser'] is True    # slow, but in the browser
    r = client.post('/api/train', json={'epochs': 20})
    assert r.status_code == 403 and '고급 기능' in r.get_json()['errors'][0]
    assert post(client, '/api/runs', upload_for({'epochs': 20})).status_code == 201  # saving browser runs works
    as_admin(client)
    r = client.post('/teacher/teachers/4/advanced', follow_redirects=True)
    assert '고급 기능을 켰습니다' in r.get_data(as_text=True)
    client.post('/logout')
    login(client, 'abc21013', '홍길동', confirm=False)
    assert 'value="decision_tree"' in client.get('/').get_data(as_text=True)
    assert post(client, '/api/prepare', HEAVY_BGD).get_json()['browser'] is False
    as_admin(client)
    assert client.post('/teacher/teachers/3/advanced').status_code == 404    # not on itself (no prefix)


def test_the_system_admin_edits_and_deletes_a_school(client):
    add_other_school(client)
    as_admin(client)
    client.post('/teacher/teachers/4/edit', data={'email': 'new@example.org', 'memo': '2027-1학기까지'})
    page = client.get('/teacher/').get_data(as_text=True)
    assert 'new@example.org' in page and '2027-1학기까지' in page and '메일로 신청' not in page
    assert json.dumps('관리자 김교사 계정과 담당 학생 2명, 실험 기록을 모두 삭제할까요?') in page   # as tojson writes it
    r = client.post('/teacher/teachers/4/delete', follow_redirects=True)
    assert '학생 2명' in r.get_data(as_text=True)
    page = client.get('/teacher/').get_data(as_text=True)
    assert '홍길동' not in page and '김하늘' in page


def test_bad_prefixes_are_refused(client):
    as_admin(client)
    for bad in ['a', 'ab1', 'abcdefghijk', '가나']:
        r = client.post('/teacher/teachers', data={'login_id': f'x{bad}', 'name': 'x', 'prefix': bad}, follow_redirects=True)
        assert '접두사는 영문자 2~10자' in r.get_data(as_text=True), bad


def test_the_example_class_list(client):
    as_admin(client)
    r = client.get('/teacher/roster-example.csv')
    assert r.get_data(as_text=True) == '﻿student_id,name\r\n21013,홍길동\r\n'
    assert '/teacher/roster-example.csv' in client.get('/teacher/').get_data(as_text=True)


def test_report_data_of_a_browser_run(client):
    login(client)
    assert post(client, '/api/runs', upload_for({'epochs': 20})).status_code == 201
    as_admin(client)
    r = client.get('/teacher/report.json')
    assert r.headers['X-Report-Size'] == str(len(r.get_data()))              # the backup's progress bar
    data = r.get_json()
    entry = next(s for s in data['students'] if s['number'] == '20101')['runs'][0]
    assert 'result' not in entry and entry['replay'] == {'trainer': 1, 'data_changed': False}
    meta = data['metas'][entry['meta']]
    assert len(meta['x_train'][0]) + len(meta['x_val'][0]) == 500 and entry['total_steps'] == 20
    assert data['teacher'] == {'name': '선생님', 'prefix': ''}
    assert client.get('/teacher/report.json?teacher=3').status_code == 200   # the system admin: any teacher's


def test_the_backup_script_is_one_classic_script(client):
    as_admin(client)
    script = client.get('/teacher/report-bundle.js').get_data(as_text=True)
    assert script == bundle(REPORT_MODULES)
    assert 'import {' not in script and '\nexport ' not in script and "modules['report-offline']" in script
    if shutil.which('node') is None:
        pytest.skip('node is not installed')
    r = subprocess.run(['node', '--check', '-'], input=script, text=True, capture_output=True)   # it parses
    assert r.returncode == 0, r.stderr


def test_the_page_offers_the_backup_and_no_runs_csv(client):
    as_admin(client)
    page = client.get('/teacher/').get_data(as_text=True)
    assert 'data-backup=""' in page and 'teacher.js' in page and 'runs.csv' not in page
    assert json.loads(page.split('window.OPTIONS = ')[1].split(';</script>')[0])['presets']


def test_delete_all_students(client):
    add_other_school(client)
    r = client.post('/teacher/students/delete-all', follow_redirects=True)      # no phrase typed: nothing goes
    assert '삭제하지 않았습니다' in r.get_data(as_text=True) and '홍길동' in r.get_data(as_text=True)
    assert client.post('/teacher/students/delete-all', data={'confirm_text': '삭 제'}).status_code == 302
    assert '홍길동' in client.get('/teacher/').get_data(as_text=True)          # a wrong phrase neither
    assert [s['number'] for s in client.get('/teacher/report.json?class=1').get_json()['students']] == ['21013', '21014']
    r = client.post('/teacher/students/delete-all', data={'confirm_text': ' Back  up and DELETE '}, follow_redirects=True)
    assert '학생 2명과 실험 기록을 삭제했습니다.' in r.get_data(as_text=True)     # kim: their own list only
    as_admin(client)
    page = client.get('/teacher/').get_data(as_text=True)
    assert '홍길동' not in page and '김하늘' in page and '학생 2명과 실험 기록을 모두 삭제합니다.' in page
    login(client)                                                               # (as 김하늘: a run to delete)
    as_admin(client)
    client.post('/teacher/students/delete-all', data={'confirm_text': '삭제'})   # the admin: the home school's students
    assert '김하늘' not in client.get('/teacher/').get_data(as_text=True)


def test_confirm_questions_are_in_single_quoted_attributes(client):
    """tojson escapes ' but not ": in a double-quoted onsubmit the question ended the attribute, the handler was
    broken and the form went without asking."""
    add_other_school(client)
    as_admin(client)
    page = client.get('/teacher/').get_data(as_text=True)
    assert 'onsubmit="' not in page and page.count("onsubmit='return confirm(") >= 4
