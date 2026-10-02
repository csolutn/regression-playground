# Machine Learning Playground

A web GUI for exploring supervised regression. Students set up the four steps of the
learning loop, press **Train**, and watch the prediction change in the browser.

```
① Training data  →  ② Prediction function  →  ③ Loss function  →  ④ Optimization
   (x, y)             ŷ = f(x)                 how errors are scored   how the graph is corrected
```

- Left pane: the four step cards; each opens its settings. The basics come first; **Advanced** holds the rest.
  - Models: linear regression and neural network are the basics; decision tree, random forest and
    gradient boosting are under ② Advanced.
  - ② shows the model structure: the network, or one tree of the chosen depth × the number of trees.
- Right pane: the newest result on top (animation with play / pause / epoch slider, the loss at that
  frame, what changed from the previous run), and the history table below it.
- **Save MP4**: a 1080×1920 video (title, prediction, loss curve) made in the browser.
  The server sends only numbers, so animation and video cost no server time.
- Login: student number + name from the teacher's class list. The first login sets the password.
- Guests (public demo, "Use as a guest" below the login form): no database. Runs train in the browser and
  stay in that page (gone on reload); tree models and runs too big for the browser need a login.
- Teacher page (`/teacher`): import the class list, reset passwords, see every student's history,
  download all runs as CSV.

## Run locally

```bash
uv sync
cp .env.example .env                                       # then set SECRET_KEY
uv run flask --app app create-teacher                      # asks for ID, name, password
uv run flask --app app import-roster students.csv          # optional; the teacher page can do this too
uv run flask run --debug                                   # http://127.0.0.1:5050
```

The class list is a CSV with `학번,이름` (or `student_id,name`) columns. The database is SQLite at
`instance/playground.db`; set `DATABASE_URL` to use PostgreSQL or MySQL instead.

**Where runs train.** Linear regression and neural network runs train in the student's browser
(`app/static/js/nn.js` in a Web Worker): the server only builds the data (`POST /api/prepare`) and
saves the finished run (`POST /api/runs`), so a class training at once costs the server almost
nothing. The numbers differ from a server run with the same seed (other random weights and batch
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

## Deploy (Mac mini + Cloudflare Tunnel)

The app runs in Docker inside a Colima VM with fixed CPU and memory, so a busy class cannot slow
down the rest of the Mac. Students reach it at `https://ml.emiclear.org` through a Cloudflare Tunnel:
the Mac opens no ports (only `127.0.0.1:8080` for the teacher on the Mac itself).

```bash
brew install colima docker docker-compose
colima start --cpu 8 --memory 8 --disk 40      # the VM's limits; 4 cores stay with macOS
brew services start colima                      # start again after a reboot
cp deploy.env.example deploy.env                # set SECRET_KEY
cp tunnel.env.example tunnel.env                # set TUNNEL_TOKEN (below)
docker compose up -d --build
docker compose exec web flask --app app create-teacher
```

Cloudflare: the `emiclear.org` zone must use Cloudflare's nameservers (when moving it, copy every
existing DNS record first, MX / SPF / DKIM included, or e-mail stops). Then Zero Trust → Networks →
Tunnels → Create (cloudflared) → copy the token into `tunnel.env` → Public hostname
`ml.emiclear.org` → service `http://web:8000`. Also: SSL/TLS → Always Use HTTPS; a rate-limiting rule
for `/login`; optionally Cloudflare Access (e-mail one-time code) on `/teacher*`.

In the container (8-CPU VM, 6 trainings at once) 30 students pressing Train together all finished within
6 s; the largest SGD run (3 × 32 neurons) takes about 43 s while others train, so `deploy.env` sets
`TRAINING_TIME_LIMIT_S=60` (measured before most runs moved to the browser; the browser uses the
same limit). One run sends 0.1–1.6 MB (2 inputs is the most).

Mac: System Settings → Energy → prevent automatic sleep and start up after a power failure.
Update with `git pull && docker compose up -d --build && docker image prune -f` (the last part deletes
the untagged images left by earlier builds; tagged ones such as a rollback image stay). Static files are served at
`/static/<hash of app/static>/…` and cached for a year, so after a deploy no browser mixes old JS with the new page. Back up with `scripts/backup.sh`
(copies the database into `backups/`, keeps 30; run it daily, e.g. from launchd).

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
| Loss landscape (one-input linear regression only) | `app/static/js/landscape.js`, `drawLandscape` in `plot.js` |
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
