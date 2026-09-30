import pytest

from app import create_app
from app.extensions import db
from app.models import User
from app.roster import import_roster


@pytest.fixture
def app(tmp_path):
    app = create_app({'TESTING': True, 'SQLALCHEMY_DATABASE_URI': f'sqlite:///{tmp_path}/test.db', 'SECRET_KEY': 'test'})
    with app.app_context():
        import_roster([('20101', '김하늘'), ('20102', '이바다')])
        teacher = User(login_id='teacher', name='선생님', role='teacher')
        teacher.set_password('teachpw')
        db.session.add(teacher)
        db.session.commit()
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def login(client, login_id='20101', name='김하늘', password='pass1234', confirm=True):
    data = {'login_id': login_id, 'name': name, 'password': password}
    if confirm:
        data['password_confirm'] = password
    return client.post('/login', data=data)
