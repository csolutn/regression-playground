# ŷ Playground

**한국어** | [English](README.en.md)

지도학습 회귀를 직접 해 보는 웹 실험실입니다. 학생이 학습의 네 단계를 정하고 **학습**을 누르면,
예측이 바뀌어 가는 모습을 브라우저에서 바로 봅니다.

**데모: <https://yhat.solsam.app>** — 게스트로 바로 열립니다. 계정이 없어도 됩니다.

```
① 훈련 데이터  →  ② 예측함수  →  ③ 손실함수           →  ④ 최적화
   (x, y)          ŷ = f(x)       오차를 어떻게 매길지      그래프를 어떻게 고칠지
```

- 왼쪽: 네 단계 카드. 카드마다 설정이 열리고, 기본 설정이 먼저, 나머지는 **고급 설정**에 있습니다.
  - 모델: 선형회귀와 인공신경망이 기본이고, 결정트리·랜덤포레스트·그래디언트 부스팅은 ② 고급 설정에 있습니다.
  - ②에서 모델 구조(신경망 그림, 또는 정한 깊이의 트리 × 트리 개수)를 보여 줍니다.
- 오른쪽: 맨 위에 최신 결과(재생 / 일시정지 / 에포크 슬라이더가 있는 애니메이션과 그 시점의 손실), 아래에
  학습 이력 표(바로 앞 실험과 달라진 칸이 강조됩니다).
- **MP4 저장**: 제목·예측·손실 곡선이 담긴 1080×1920 영상을 브라우저에서 만듭니다.
  서버는 숫자만 보내므로 애니메이션과 영상에 서버 시간이 들지 않습니다.
- 게스트: 첫 화면(로그아웃한 뒤에도)은 게스트용 실험실이고 데이터베이스를 쓰지 않습니다. 학습은 브라우저에서
  하고 결과는 그 페이지에만 남습니다(새로고침하면 사라짐). 트리 모델과 브라우저에서 돌리기에 너무 큰 학습은
  로그인해야 합니다.
- 로그인: 위쪽 바의 **로그인**이 팝업을 엽니다(로그인 페이지는 따로 없고, `/login`과 로그인이 필요한 페이지는
  `/?login=1`로 팝업을 엽니다). 교사가 올린 명단의 학번 + 이름으로 들어오고, 처음 로그인할 때 비밀번호를
  정합니다. 게스트로 한 실험은 그 계정에 저장됩니다.
- 교사 페이지(`/teacher`): 학생 명단 불러오기, 비밀번호 초기화, 학생별 학습 이력, 전체 실험 CSV 내려받기.
  가장 먼저 만든 교사 계정은 **시스템 관리자**로, 다른 교사 계정을 추가·초기화·삭제할 수 있습니다(아이디와
  이름만 넣고, 비밀번호는 처음 로그인할 때 정합니다).

## 준비물

- 내 컴퓨터에서 실행: [uv](https://docs.astral.sh/uv/) (Python 3.12와 패키지를 알아서 설치합니다).
- 서버 배포: Docker와 Docker Compose 2.24 이상(`docker compose version`). 이미지는 약 2 GB(CPU용 torch),
  실행 중 메모리는 약 1 GB입니다.
- 브라우저 학습 테스트: Node 20 이상.

## 내 컴퓨터에서 실행

```bash
uv sync
cp .env.example .env                                       # SECRET_KEY를 바꾸세요
uv run flask --app app create-teacher --login-id teacher --name 선생님 --password change-me   # 또는 .env의 ADMIN_*
uv run flask --app app import-roster students.csv          # 선택. 교사 페이지에서도 됩니다
uv run flask run --debug                                   # http://127.0.0.1:15001 (포트는 .flaskenv)
```

`create-teacher`에 세 옵션을 빼면 하나씩 물어봅니다.
학생 명단은 `학번,이름`(또는 `student_id,name`) 열이 있는 CSV입니다. 데이터베이스는 `instance/playground.db`의
SQLite이고, PostgreSQL이나 MySQL을 쓰려면 `DATABASE_URL`을 정하세요.

**학습은 어디서 하나.** 선형회귀와 인공신경망은 학생 브라우저에서 학습합니다(`app/static/js/nn.js`, Web Worker).
서버는 데이터를 만들어 주고(`POST /api/prepare`) 끝난 결과를 저장할 뿐이라(`POST /api/runs`, 수 kB), 한 반이
동시에 학습해도 서버 부담이 거의 없습니다. 트리 모델과, 브라우저로는 너무 느린 큰 학습만 서버에서 합니다
(`POST /api/train`, 결정은 `app/ml/options.py`의 `trains_in_browser()`). 자세한 내용은 [English README](README.en.md)에 있습니다.

| 설정 (환경변수) | 기본값 | 뜻 |
|---|---|---|
| `MAX_CONCURRENT_TRAININGS` | 6 (코어가 적으면 그만큼) | 서버에서 동시에 학습하는 프로세스 수. 나머지는 줄을 섭니다 |
| `QUEUE_TIMEOUT_S` | 90 | 이만큼 기다려도 자리가 없으면 "서버가 바쁩니다"라고 알립니다 |
| `TRAINING_TIME_LIMIT_S` | 30 | 학습이 이보다 길어지면 멈추고 "시간 제한"으로 저장합니다 |

## 배포 (Docker + Cloudflare Tunnel, 어떤 도메인이든)

Docker를 돌릴 수 있는 컴퓨터(리눅스 서버, 또는 [Colima](https://github.com/abiosoft/colima)나 Docker Desktop을
쓰는 Mac)에서 CPU·메모리 한도를 정해 실행합니다. 학생들은 내 도메인(예: `playground.example.com`)으로
Cloudflare Tunnel을 거쳐 들어오므로 컴퓨터는 포트를 열지 않습니다(확인용으로 그 컴퓨터의 `127.0.0.1:8000`만 엽니다).

**체크리스트.** *(사람)* 표시는 사람이 해야 하는 일(대시보드 작업, 결정)이고, 나머지는 명령입니다.

1. *(사람)* 주소를 정하고, Cloudflare Tunnel(아래, 도메인이 Cloudflare에 있어야 함), 직접 운영하는 HTTPS 프록시,
   학교망 안의 일반 http(HTTPS 없음, "Cloudflare 없이" 참고) 중 하나를 고릅니다.
2. `cp deploy.env.example deploy.env` 후 `SECRET_KEY`(만드는 명령이 파일에 있음),
   `ADMIN_LOGIN_ID` / `ADMIN_NAME` / `ADMIN_PASSWORD`, `TRUST_PROXY`(HTTPS 뒤면 1, 일반 http면 0)를 채웁니다.
3. 터널을 쓰면: *(사람)* 터널을 만들고 토큰을 복사한 뒤(아래 **Cloudflare**),
   `cp tunnel.env.example tunnel.env` 하고 `TUNNEL_TOKEN`을 넣습니다.
4. `docker compose up -d --build` (터널 없이: `docker compose up -d --build web`).
5. 확인: 약 20초 뒤 `docker compose ps`에서 web이 `(healthy)`이고,
   `curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/`이 `200`을 출력합니다. 터널을 쓰면
   `https://<주소>/`도 같은지 봅니다. 문제가 있으면 `docker compose logs web`(또는 `tunnel`). 토큰이 없거나
   틀린 터널은 계속 재시작하지만 web에는 영향이 없습니다.
6. *(사람)* 사이트를 열고 **로그인**에서 `ADMIN_*` 값으로 들어간 뒤 **교사**에서 학생 명단을 불러옵니다.

**교사 계정.** 처음 시작할 때 `deploy.env`의 `ADMIN_LOGIN_ID`, `ADMIN_NAME`, `ADMIN_PASSWORD`로 교사 계정을
만듭니다(교사 계정이 하나도 없을 때만. 그 뒤로는 아무것도 바꾸지 않으므로 이 줄은 지워도 됩니다). 이 계정이
시스템 관리자이고, 교사 페이지에서 다른 교사를 추가합니다. 교사가 비밀번호를 잊으면 시스템 관리자가 교사
페이지에서 초기화하거나, `docker compose exec web flask --app app create-teacher`에 같은 아이디를 넣어 새로 정합니다.

**크레딧.** 실험실 아래 줄(`© …`, GitHub·메일 아이콘)은 `deploy.env`의 `CREDIT_NAME`, `CREDIT_URL`,
`GITHUB_URL`, `CONTACT_EMAIL`에서 옵니다. 비어 있는 것은 표시하지 않습니다.

**Cloudflare.** 도메인이 Cloudflare에 있어야 합니다(네임서버. 다른 곳에서 옮겨 오면 MX / SPF / DKIM을 포함한
DNS 레코드를 먼저 모두 옮겨야 메일이 끊기지 않습니다). Zero Trust → Networks → Tunnels → Create (cloudflared) →
토큰을 `tunnel.env`에 복사 → Public hostname `playground.example.com` → service `http://web:8000`.
추가로: SSL/TLS → Always Use HTTPS, `POST /login` 속도 제한 규칙(예: IP당 10초에 300번, 10초 차단. 한 학교가
IP 하나를 같이 쓰므로 한 반이 동시에 로그인해도 걸리지 않고 폭주만 막습니다), 원하면 `/teacher*`에
Cloudflare Access(메일 일회용 코드).

**Cloudflare 없이.** web만 띄우고(`docker compose up -d --build web`, `tunnel.env` 필요 없음)
`compose.override.yaml`(아래)에서 포트를 네트워크에 엽니다(예: `"0.0.0.0:8000:8000"`). `X-Forwarded-Proto`를
넘겨 주는 HTTPS 리버스 프록시(Caddy, nginx) 뒤라면 `TRUST_PROXY=1` 그대로 둡니다. 학교망에서
`http://<이 컴퓨터>:8000`으로 쓰면 `TRUST_PROXY=0`으로 하세요. 1이면 브라우저가 로그인 쿠키를 HTTPS에서만
보내서 로그인과 학습이 실패합니다.

**이 컴퓨터만의 설정**(다른 포트, CPU 수)은 git에 들어가지 않는 `compose.override.yaml`에 둡니다.
`docker compose`가 `compose.yaml` 위에 덧씌워 읽습니다. 예:

```yaml
services:
  web:
    ports: !override
      - "127.0.0.1:18001:8000"
    cpus: 3
```

`cpus`는 Docker가 가진 CPU 수를 넘을 수 없습니다(Mac이면 Colima VM의 수: `colima start --cpu 4 --memory 4`).
8-CPU VM에서 동시 학습 6개로 학생 30명이 한꺼번에 학습을 눌러도 6초 안에 모두 끝났습니다(대부분의 학습을
브라우저로 옮기기 전 측정).

정전 뒤 컴퓨터가 다시 켜지고 Docker가 부팅 때 시작되게 하세요(Mac의 Colima: `brew services start colima`).
업데이트: `git pull && docker compose up -d --build && docker image prune -f`(마지막은 이전 빌드가 남긴 이름 없는
이미지를 지웁니다. 롤백용처럼 태그를 단 이미지는 남습니다). 백업: `scripts/backup.sh`(데이터베이스를 `backups/`나
`BACKUP_DIR`(예: 동기화되는 클라우드 폴더)에 복사, 30개 보관. cron이나 launchd로 매일 실행).

## 개발

코드 구조("어디를 고치면 되나"), 번역(Babel), 학습 방식의 자세한 설명은 [English README](README.en.md)에 있습니다.
테스트:

```bash
uv run pytest
node --test tests/js          # 브라우저 학습 (app/static/js/nn.js)
```

## 라이선스

MIT (`LICENSE` 참고). 함께 들어 있는 다른 사람의 파일은 각자의 라이선스를 따릅니다: Grandstander 글꼴
(`app/static/fonts/`, SIL Open Font License 1.1, `OFL-Grandstander.txt`)과
[mp4-muxer](https://github.com/Vanilagy/mp4-muxer) (`app/static/vendor/`, MIT).
