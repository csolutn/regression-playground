// Results of a student's runs without the animation: for the run picked in the history table, the prediction it
// ended with (its last frame) and its loss curve. Used by the teacher's student page (report-live.js, which trains
// browser runs again to draw them) and by the HTML backup (report-offline.js, everything drawn already). Nothing
// here talks to the server: bundle.py puts this file into the backup.
//
// The data (from /teacher/report.json, see app/report.py): { students: [{ prefix, number, name, runs: [entry] }],
// metas: [the data a run learned from] }; entry = { row, meta: index in metas, total_steps,
// result?: { frame, loss, note? }, replay?: { trainer, data_changed }, error?: true }
import { t } from './i18n.js';
import { drawLoss, drawPrediction, fitCanvas, fmtLoss, screenTheme } from './plot.js';
import { History, esc } from './history.js';
import { dataSummary, lossSummary, modelSummary, optimSummary } from './describe.js';

const MAX_LOSS_POINTS = 200;
const round = v => (v == null || !Number.isFinite(v) ? v : Number(v.toPrecision(4)));

// a run as plot.js draws it: its data with the run's own fields, and its last frame
export function drawable(entry, metas) {
  const config = entry.row.config, r = entry.result;
  const meta = { ...metas[entry.meta], model: config.model, loss: config.loss, total_steps: entry.total_steps };
  return { config, row: entry.row, meta, frames: r?.frame ? [r.frame] : [], loss: r?.loss || { steps: [], train: [], val: [] } };
}

// what a replay (train.js replayRun) made, kept small: the last frame and about 200 points of the loss curve
export function resultOf(made, note = null) {
  const last = made.frames.at(-1);
  const frame = last && { step: last.step, pred: last.pred.map(round), train: round(last.train), val: round(last.val) };
  const n = made.loss.steps.length, k = Math.max(1, Math.ceil(n / MAX_LOSS_POINTS));
  const keep = [];
  for (let i = 0; i < n; i += k) keep.push(i);
  if (n && keep.at(-1) !== n - 1) keep.push(n - 1);
  const loss = Object.fromEntries(['steps', 'train', 'val'].map(s => [s, keep.map(i => round(made.loss[s][i]))]));
  return note ? { frame, loss, note } : { frame, loss };
}

export function idLabel(student) {
  return `${student.prefix ? `<span class="id-prefix">${esc(student.prefix)}</span>` : ''}${esc(student.number)}`;
}

export class Report {
  // resolve(entry, onProgress, signal) → its result (report-live.js trains it again; the backup has it already)
  constructor(root, { metas, resolve }) {
    this.metas = metas;
    this.resolve = resolve;
    root.innerHTML = `
      <article class="card result-card report-result">
        <header class="result-head">
          <div class="result-titles"><h2 data-r="title"></h2><p data-r="sub" class="muted"></p></div>
          <span data-r="status"></span>
        </header>
        <div class="report-plots">
          <div class="viz">
            <canvas data-r="pred" aria-label="${esc(t('Prediction'))}"></canvas>
            <div data-r="empty" class="viz-empty"></div>
            <div data-r="busy" class="viz-busy" hidden>
              <div class="viz-busy-box"><span data-r="busy-text"></span><div class="progress"><div data-r="bar"></div></div></div>
            </div>
          </div>
          <canvas data-r="loss" class="report-loss" aria-label="${esc(t('Loss graph'))}"></canvas>
        </div>
        <div class="readout">
          <span><span class="dot dot-train"></span>${esc(t('Training loss'))} <b data-r="train">—</b></span>
          <span><span class="dot dot-val"></span>${esc(t('Validation loss'))} <b data-r="val">—</b></span>
        </div>
        <p data-r="note" class="note" hidden></p>
      </article>
      <article class="card history-card">
        <header class="history-head">
          <div class="history-titles">
            <h2>${esc(t('History'))}</h2>
            <span class="muted">${esc(t('Highlighted cells changed from the run before. Click a row to show it.'))}</span>
          </div>
          <div class="history-tools">
            <span class="muted" data-r="count"></span>
            <button type="button" class="btn btn-ghost btn-sm" data-r="csv">${esc(t('Save CSV'))}</button>
          </div>
        </header>
        <div class="table-wrap"><table data-r="history" class="history"></table></div>
        <div data-r="pop" class="filter-pop" role="dialog" hidden></div>
      </article>`;
    const $ = name => root.querySelector(`[data-r="${name}"]`);
    this.el = Object.fromEntries(['title', 'sub', 'status', 'pred', 'empty', 'busy', 'busy-text', 'bar', 'loss',
      'train', 'val', 'note', 'count', 'csv'].map(n => [n, $(n)]));
    this.history = new History($('history'), { readonly: true, popup: $('pop'), onView: row => this.view(row) });
    this.history.onRender = (shown, total) => {
      this.el.count.textContent = this.history.filtered ? t('{n} of {total} runs', { n: shown, total }) : '';
      this.el.csv.disabled = shown === 0;
    };
    this.el.csv.addEventListener('click', () => {
      const s = this.student, blob = new Blob([this.history.toCsv()], { type: 'text/csv;charset=utf-8' });
      const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: `history_${s.prefix}${s.number}.csv` });
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    });
    this.run = null;
    this.pending = null;              // the AbortController of the run being trained again
    for (const c of [this.el.pred, this.el.loss]) new ResizeObserver(() => this.render()).observe(c);
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.render());
  }

  setStudent(student) {
    this.student = student;
    this.entries = new Map(student.runs.map(e => [e.row.id, e]));
    this.history.setRows(student.runs.map(e => e.row));
    if (student.runs.length) { this.view(student.runs[0].row); return; }
    this.pending?.abort();
    for (const k of ['sub', 'status', 'note']) this.el[k].textContent = '';
    this.el.status.className = '';
    this.el.title.textContent = t('Result');
    this.el.train.textContent = this.el.val.textContent = '—';
    this.show(null, t('No runs yet.'));
  }

  async view(row) {
    const entry = this.entries.get(row.id);
    this.pending?.abort();
    const ac = this.pending = new AbortController();
    this.history.setActive(row.id);
    this.header(row);
    if (entry.error) { this.show(null, t('The data of this run can not be made again.')); return; }
    if (!entry.result) this.show(entry, '', { dataOnly: true });     // its data while it is trained again
    try {
      entry.result ||= await this.resolve(entry, p => this.progress(p), ac.signal);
      if (this.pending === ac) this.show(entry);
    } catch (err) {
      if (err.name !== 'AbortError') this.show(null, err.message);
    } finally {
      if (this.pending === ac) { this.pending = null; this.progress(null); }
    }
  }

  header(row) {
    const cfg = row.config, { title, sub, status, train, val } = this.el;
    title.textContent = `#${row.seq} · ${modelSummary(cfg)}`;
    sub.textContent = [dataSummary(cfg), lossSummary(cfg), optimSummary(cfg)].join('  |  ');
    const st = { timeout: ['warn', t('Stopped at the time limit')], diverged: ['warn', t('Diverged: the loss became infinite. Try a smaller learning rate.')] }[row.status]
      || ['ok', t('Done')];
    status.className = `pill pill-${st[0]}`;
    status.textContent = st[1];
    train.textContent = fmtLoss(row.final_train);
    val.textContent = fmtLoss(row.final_val);
  }

  show(entry, message = '', { dataOnly = false } = {}) {
    this.run = entry ? drawable(entry, this.metas) : null;
    this.dataOnly = dataOnly;
    this.el.empty.textContent = message;
    this.el.empty.hidden = !!entry;
    this.el.note.hidden = !entry?.result?.note;
    this.el.note.textContent = entry?.result?.note || '';
    this.render();
  }

  progress(p) {
    this.el.busy.hidden = p == null;
    if (p == null) return;
    const pct = Math.round(Math.min(1, p) * 100);
    this.el.bar.style.width = `${pct}%`;
    this.el['busy-text'].textContent = `${t('Rebuilding the training…')} ${pct}%`;
  }

  render() {
    const th = screenTheme();
    for (const [canvas, draw] of [[this.el.pred, (ctx, r) => drawPrediction(ctx, r, this.run, 0, th, 1, { dataOnly: this.dataOnly })],
                                  [this.el.loss, (ctx, r) => drawLoss(ctx, r, this.run, Infinity, th, 1)]]) {
      const { ctx, w, h } = fitCanvas(canvas);
      ctx.fillStyle = th.bg;
      ctx.fillRect(0, 0, w, h);
      if (this.run && !(this.dataOnly && canvas === this.el.loss)) draw(ctx, { x: 0, y: 0, w, h });
    }
  }
}
