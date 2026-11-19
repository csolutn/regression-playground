# ŷ Playground

[한국어](README.md) | **English**

A web GUI for exploring supervised regression. Students set up the four steps of the
learning loop, press **Train**, and watch the prediction change in the browser.

**Demo: <https://yhat.solsam.app>** — opens as a guest; no account needed.

```
① Training data  →  ② Predictor  →  ③ Loss function        →  ④ Optimization
   (x, y)             ŷ = f(x)        how errors are scored     how the graph is corrected
```

- Left pane: the four step cards; each opens its settings. The basics come first; **Advanced** holds the rest.
  - Models: linear regression and neural network are the basics; decision tree, random forest and
    gradient boosting are under ② Advanced.
  - ② shows the model structure: the network, or one tree of the chosen depth × the number of trees.
- Right pane: the newest result on top (animation with play / pause / epoch slider, the loss at that
  frame), and the history table below it (cells that changed from the run before are highlighted).
- **Save MP4**: a 1080×1920 video (title, prediction, loss curve) made in the browser.
  The server sends only numbers, so animation and video cost no server time.
- Guests: the front page (also after logging out) is the playground as a guest, with no database. Runs
  train in the browser and stay in that page (gone on reload); tree models and runs too big for the
  browser need a login.
- Login: **Log in** in the top bar opens a popup (there is no login page; `/login` and pages that need a
  login open it at `/?login=1`). Student number + name from the teacher's class list; the first login
  sets the password. The runs made as a guest are saved to the account.
- Teacher page (`/teacher`): import the class list, reset passwords, see every student's history,
  download all runs as CSV. The first teacher account is the system admin: it also adds, resets and deletes the other
  teachers (login ID and name; the first login sets the password).

## Requirements

- Running locally: [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 and the packages).
- Deploying: Docker with Docker Compose 2.24 or newer (`docker compose version`). The image is
  about 2 GB (CPU-only torch); the running app uses about 1 GB of memory.
- The browser tests: Node 20 or newer.

## Run locally

```bash
uv sync
cp .env.example .env                                       # then set SECRET_KEY
uv run flask --app app create-teacher --login-id teacher --name Teacher --password change-me   # or ADMIN_* in .env
uv run flask --app app import-roster students.csv          # optional; the teacher page can do this too
uv run flask run --debug                                   # http://127.0.0.1:15001 (port from .flaskenv)
```

Without the three options, `create-teacher` asks for them.

The class list is a CSV with `학번,이름` (or `student_id,name`) columns. The database is SQLite at
`instance/playground.db`; set `DATABASE_URL` to use PostgreSQL or MySQL instead.

**Where runs train.** Linear regression and neural network runs train in the student's browser
(`app/static/js/nn.js` in a Web Worker): the server only builds the data (`POST /api/prepare`) and
saves the finished run's results (`POST /api/runs`, a few kB), so a class training at once costs the
server almost nothing. The animation, loss curve and loss landscape are not stored: when a saved run is
shown again, the browser trains it again from its settings and seed, which gives the same numbers
(`TRAINER_VERSION` in `nn.js` and `DATA_VERSION` in `app/ml/data.py` flag runs saved before a change). The numbers differ from a server run with the same seed (other random weights and batch
order), the behaviour does not. Runs that plain JavaScript would be much slower at (big networks
with big batches, which torch finishes in under a second) and the tree models train on the server
(`POST /api/train`); `trains_in_browser()` in `app/ml/options.py` decides.

For a classroom server (about 30 students at once), run **one** server process with enough threads
for every open training stream, for example
`uv run --with gunicorn gunicorn -w 1 --threads 40 -b 0.0.0.0:5050 "app:create_app()"`.
Training itself runs in worker processes started by that one process (`app/ml/pool.py`), because
runs in threads of one process slow each other down badly (the GIL). More gunicorn workers would
each start their own pool and oversubscribe the CPU.

| Setting (env) | Default | Meaning |
|---|---|---|
| `MAX_CONCURRENT_TRAININGS` | 6 (or fewer cores) | worker processes training at once; other runs wait in line |
| `QUEUE_TIMEOUT_S` | 90 | after waiting this long for a free worker, the student is told the server is busy |
| `TRAINING_TIME_LIMIT_S` | 30 | a run is stopped after this long and saved with status "time limit" |

On a 12-core Mac, 30 students pressing Train at the same moment all finished within 5 s; with
every fifth student running the largest allowed SGD run, the others still started within 7 s.

## Deploy (Docker + Cloudflare Tunnel, any domain)

The app runs in Docker with fixed CPU and memory limits, on any machine that can run Docker (a Linux
server, or a Mac with [Colima](https://github.com/abiosoft/colima) or Docker Desktop). Students reach it at
your own hostname, say `playground.example.com`, through a Cloudflare Tunnel: the machine opens no ports
(only `127.0.0.1:8000` on the machine itself, for checks).

**Checklist.** Steps marked *(you)* need a person (a dashboard, a decision); the rest are commands.

1. *(you)* Pick the hostname, and decide: Cloudflare Tunnel (below, needs the domain on Cloudflare) or
   your own HTTPS proxy, or plain http on a school network (no HTTPS; see "Without Cloudflare").
2. `cp deploy.env.example deploy.env`, then set `SECRET_KEY` (the command is in the file),
   `ADMIN_LOGIN_ID` / `ADMIN_NAME` / `ADMIN_PASSWORD`, and `TRUST_PROXY` (1 behind HTTPS, 0 for plain http).
3. With the tunnel: *(you)* create it and copy its token (see **Cloudflare** below), then
   `cp tunnel.env.example tunnel.env` and set `TUNNEL_TOKEN`.
4. `docker compose up -d --build` (without a tunnel: `docker compose up -d --build web`).
5. Check: `docker compose ps` shows the web service `(healthy)` after about 20 s, and
   `curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/` prints `200`; with the tunnel, the
   same for `https://<hostname>/`. Problems: `docker compose logs web` (or `tunnel`). A tunnel without a
   valid token restarts over and over; the web service is not affected.
6. *(you)* Open the site, **Log in** with the `ADMIN_*` values, go to **Teacher** and import the class list.

**The teacher account.** At its first start the app makes a teacher account from `ADMIN_LOGIN_ID`,
`ADMIN_NAME` and `ADMIN_PASSWORD` in `deploy.env` (only while there is no teacher yet; nothing changes
later, so the lines may be deleted afterwards). Log in with them from **Log in** in the top bar and open
**Teacher** (`/teacher`) to import the class list and add other teachers. A teacher's forgotten password:
`docker compose exec web flask --app app create-teacher` (the same login ID sets a new password).

**Credits.** The line under the playground (`© …`, GitHub and e-mail icons) comes from `CREDIT_NAME`,
`CREDIT_URL`, `GITHUB_URL` and `CONTACT_EMAIL` in `deploy.env`; empty ones are left out.

**Cloudflare.** Your domain must be on Cloudflare (its nameservers; a domain moved from elsewhere needs
every DNS record copied first, MX / SPF / DKIM included, or e-mail stops). Then Zero Trust → Networks →
Tunnels → Create (cloudflared) → copy the token into `tunnel.env` → Public hostname
`playground.example.com` → service `http://web:8000`. Also: SSL/TLS → Always Use HTTPS; a rate-limiting
rule for `POST /login` (e.g. 300 requests / 10 s per IP, block 10 s: a whole school shares one IP, so it
only catches floods, never a class logging in); optionally Cloudflare Access (e-mail one-time code) on `/teacher*`.

**Without Cloudflare.** Start only the web service (`docker compose up -d --build web`; `tunnel.env` is
not needed) and publish its port to the network in `compose.override.yaml` (below), e.g.
`"0.0.0.0:8000:8000"`. Behind your own HTTPS reverse proxy (Caddy, nginx) that sets `X-Forwarded-Proto`,
keep `TRUST_PROXY=1`. For plain `http://<this machine>:8000` on a school network set `TRUST_PROXY=0`:
with 1 the browser keeps the login cookie for HTTPS only, so logging in and training fail.

**One machine's own settings** (another host port, more CPUs) go in `compose.override.yaml`, which is
not in git and which `docker compose` reads on top of `compose.yaml`, for example

```yaml
services:
  web:
    ports: !override
      - "127.0.0.1:18001:8000"
    cpus: 3
```

`cpus` may not exceed the CPUs Docker has (on a Mac, the Colima VM's: `colima start --cpu 4 --memory 4`).
In an 8-CPU VM with 6 trainings at once, 30 students pressing Train together all finished within 6 s; the
largest SGD run (3 × 32 neurons) takes about 43 s while others train, so `deploy.env` sets
`TRAINING_TIME_LIMIT_S=60` (measured before most runs moved to the browser; the browser uses the same
limit). A server run streams 0.1–1.6 MB (2 inputs is the most); a browser run uploads a few kB.

Let the machine start again after a power failure, and start Docker at boot (Colima on a Mac:
`brew services start colima`). Update with `git pull && docker compose up -d --build && docker image prune -f`
(the last part deletes the untagged images left by earlier builds; tagged ones such as a rollback image stay).
Static files are served at `/static/<hash of app/static>/…` and cached for a year, so after a deploy no
browser mixes old JS with the new page. Back up with `scripts/backup.sh` (copies the database into
`backups/`, or `BACKUP_DIR` if set, e.g. a synced cloud folder; keeps 30; run it daily, e.g. from cron or launchd).

## Where to change things

| To change | Edit |
|---|---|
| A setting (default, range, choices) | `app/ml/options.py` — the single list of settings and limits |
| Its input on screen | the step template in `app/templates/steps/` (`data`, `model`, `loss`, `optim`) — inputs are bound by `name` |
| Preset functions | `PRESETS_1D` / `PRESETS_2D` in `app/ml/data.py` |
| Training (models, optimizers, losses) | `app/ml/trainer.py` — a plain generator of events, no Flask; linear and neural network runs also in `app/static/js/nn.js` (the browser), keep both in step |
| Which runs train in the browser | `trains_in_browser()` in `app/ml/options.py` |
| Summaries in cards, history, badges, video title | `app/static/js/describe.js` |
| Plot drawing (screen and video) | `app/static/js/plot.js` |
| Loss landscape: the whole surface (one-input linear regression), else slices along the 4 weights that lowered the loss most (dashed: the landscape at the end), each with the slope that weight came down while training (solid: its tangents joined up; the point moves with the epoch slider) | `app/static/js/landscape.js`, `drawLandscape` / `drawSlices` in `plot.js`; the weights' contributions are added up while training (`train()` in `nn.js`, `Contributions` in `trainer.py`) |
| Player (play / pause / slider) | `app/static/js/player.js` |
| Video layout and size | `app/static/js/recorder.js` (`W`, `H` at the top) |
| History columns, change highlighting, filters (date, data) and ★ | `app/static/js/history.js`; what counts as the same data: `dataKey` in `describe.js` |
| App name | `APP_NAME` in `app/config.py` |
| Database tables | `app/models.py` |

## Translations (Babel)

Source strings are English; Korean is in `app/translations/ko/LC_MESSAGES/messages.po`.
JavaScript uses `t('English text')` and gets the same catalog. After changing any text:

```bash
uv run pybabel extract -F babel.cfg -k t -k DataError -k N_ --no-location --sort-output -o app/translations/messages.pot .
uv run pybabel update -i app/translations/messages.pot -d app/translations --no-fuzzy-matching
# fill the new msgstr entries in messages.po, then
uv run pybabel compile -d app/translations
```

In JavaScript, call `t()` outside template literals when it has a `{…}` parameter object,
so that `pybabel extract` finds it.

## Tests

```bash
uv run pytest
node --test tests/js          # the browser trainer (app/static/js/nn.js)
```

## License

MIT (see `LICENSE`). Bundled third-party files keep their own licenses: the Grandstander font
(`app/static/fonts/`, SIL Open Font License 1.1, `OFL-Grandstander.txt`) and
[mp4-muxer](https://github.com/Vanilagy/mp4-muxer) (`app/static/vendor/`, MIT).
