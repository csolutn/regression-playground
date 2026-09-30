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

For a classroom server, run it with a threaded server, for example
`uv run --with gunicorn gunicorn -w 2 --threads 8 "app:create_app()"`.
`MAX_CONCURRENT_TRAININGS` (default 4) limits how many runs train at once; the rest wait in line.

## Where to change things

| To change | Edit |
|---|---|
| A setting (default, range, choices) | `app/ml/options.py` — the single list of settings and limits |
| Its input on screen | the step template in `app/templates/steps/` (`data`, `model`, `loss`, `optim`) — inputs are bound by `name` |
| Preset functions | `PRESETS_1D` / `PRESETS_2D` in `app/ml/data.py` |
| Training (models, optimizers, losses) | `app/ml/trainer.py` — a plain generator of events, no Flask |
| Summaries in cards, history, badges, video title | `app/static/js/describe.js` |
| Plot drawing (screen and video) | `app/static/js/plot.js` |
| Player (play / pause / slider) | `app/static/js/player.js` |
| Video layout and size | `app/static/js/recorder.js` (`W`, `H` at the top) |
| History columns and change highlighting | `app/static/js/history.js` |
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
```
