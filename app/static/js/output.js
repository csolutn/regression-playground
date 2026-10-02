// Right-hand output: result header, animation player, MP4 export and history table.
// Used by the playground (students) and, read-only, by the teacher's student page.
import { t } from './i18n.js';
import { api } from './api.js';
import { Player } from './player.js';
import { History, diffSummary, esc, localDate } from './history.js';
import { exportVideo, saveBlob } from './recorder.js';
import { fmtLoss } from './plot.js';
import { dataSummary, lossSummary, modelSummary, optimSummary, videoTitle } from './describe.js';

export class OutputPanel {
  constructor(root, { readonly = false, userId = null, toast, onLoadSettings } = {}) {
    const $ = role => root.querySelector(`[data-role="${role}"]`);
    this.el = {
      title: $('result-title'), sub: $('result-sub'), status: $('result-status'), diff: $('result-diff'),
      busy: $('viz-busy'), busyText: $('busy-text'), bar: root.querySelector('[data-role="progress"] > div'),
      mp4: $('mp4'), dialog: $('mp4-dialog'),
      count: $('history-count'), csv: $('csv'),
    };
    this.login = root.dataset.login || '';
    this.userId = userId;
    this.toast = toast;
    this.cache = new Map();                   // run id → full run (frames etc.)
    this.player = new Player(root);
    this.history = new History($('history'), {
      readonly,
      popup: $('filter-pop'),
      onView: row => this.view(row),
      onLoad: async row => onLoadSettings?.(await this.fetchRun(row)),
      onDelete: row => this.remove(row),
    });
    this.el.mp4.addEventListener('click', () => this.openMp4Dialog());
    this.setupMp4Dialog();
    this.setupHistoryTools();
    this.showHeader(null);
  }

  get run() { return this.player.run; }

  async init() {
    try {
      const rows = await api.runs(this.userId);
      this.history.setRows(rows);
      if (rows[0]) await this.view(rows[0]);
    } catch (err) {
      this.toast?.(err.message, 'error');
    }
  }

  async fetchRun(row) {
    if (!this.cache.has(row.id)) {
      const d = await api.run(row.id);
      this.cache.set(row.id, { row: d.run, config: d.config, meta: d.meta, frames: d.frames, loss: d.loss });
    }
    return this.cache.get(row.id);
  }

  async view(row) {
    if (this.run?.live) { this.toast?.(t('Wait until training finishes.'), 'info'); return; }
    try {
      const run = await this.fetchRun(row);
      this.player.setRun(run);
      this.history.setActive(row.id);
      this.showHeader(run);
    } catch (err) {
      this.toast?.(err.message, 'error');
    }
  }

  async remove(row) {
    if (!confirm(t('Delete run #{n}? This cannot be undone.', { n: row.seq }))) return;
    try {
      await api.deleteRun(row.id);
      this.cache.delete(row.id);
      this.history.remove(row.id);
      if (this.run?.row?.id === row.id) {
        this.player.setRun(null);
        this.showHeader(null);
      }
    } catch (err) {
      this.toast?.(err.message, 'error');
    }
  }

  // ---------- history: filter count and CSV (the filters themselves are in the table head) ----------

  setupHistoryTools() {
    const { count, csv } = this.el;
    this.history.onRender = (shown, total) => {
      count.textContent = this.history.filtered ? t('{n} of {total} runs', { n: shown, total }) : '';
      csv.disabled = shown === 0;
    };
    csv.addEventListener('click', () => {
      const { from: a, to: b } = this.history.filter;
      const span = a || b ? `_${a || 'start'}_${b || localDate(new Date())}` : '';
      const blob = new Blob([this.history.toCsv()], { type: 'text/csv;charset=utf-8' });
      saveBlob(blob, `history_${this.login || 'runs'}${span}.csv`);
    });
    this.history.render();
  }

  // ---------- data preview: the data the next run will learn from, before training ----------

  showPreview(config, meta, error = null) {
    const run = { preview: true, config, meta, error, frames: [], loss: { steps: [], train: [], val: [] }, row: null };
    this.player.setRun(run);
    this.history.setActive(null);
    this.showHeader(run);
  }

  // ---------- live training ----------

  startLive(config) {
    const run = { live: true, config, meta: null, frames: [], loss: { steps: [], train: [], val: [] }, row: null };
    this.player.setRun(run, { live: true });
    this.history.setActive(null);
    this.showHeader(run);
    this.setProgress(0);
    return run;
  }

  liveEvent(run, ev) {
    if (ev.type === 'queued') {
      this.el.status.className = 'pill';
      this.el.status.textContent = t('Waiting for a free slot…');
      this.setProgress(0, t('Waiting for a free slot…'));
    } else if (ev.type === 'start') {
      run.meta = ev.meta;
      this.player.setRun(run, { live: true });
      this.showHeader(run);
    } else if (ev.type === 'frame') {
      run.frames.push(ev);
      this.player.update();
      this.setProgress(ev.step / run.meta.total_steps);
    } else if (ev.type === 'loss') {
      for (const k of ['steps', 'train', 'val']) run.loss[k].push(...ev[k]);
    } else if (ev.type === 'end') {
      run.end = ev;
    } else if (ev.type === 'error') {
      this.stopLive(run, t('Error'));
      this.toast?.(ev.message, 'error');
    } else if (ev.type === 'saved') {
      run.live = false;
      run.row = ev.run;
      this.cache.set(ev.run.id, run);
      this.history.add(ev.run);
      this.history.setActive(ev.run.id);
      this.player.endLive();
      this.setProgress(null);
      this.showHeader(run);
      this.player.playFromStart();          // the run is done: replay it from epoch 0 at the chosen speed
    }
  }

  stopLive(run, message) {
    run.live = false;
    this.player.endLive();
    this.setProgress(null);
    this.showHeader(run, message);
  }

  // progress bar over the graph while training (p = 0…1, null hides it)
  setProgress(p, text) {
    const { busy, busyText, bar } = this.el;
    busy.hidden = p == null;
    if (p == null) return;
    const pct = Math.round(Math.min(1, p) * 100);
    bar.style.width = `${pct}%`;
    busyText.textContent = text || `${t('Training…')} ${pct}%`;
  }

  // ---------- header: what this run is, and what changed from the one before ----------

  showHeader(run, message) {
    const { title, sub, status, diff, mp4 } = this.el;
    mp4.disabled = !run?.meta || run.live || run.frames.length < 2;
    if (!run) {
      title.textContent = t('Result');
      sub.textContent = t('Set up the experiment on the left and press Train.');
      status.textContent = ''; status.className = '';
      diff.innerHTML = '';
      return;
    }
    const cfg = run.config;
    if (run.preview) {
      title.textContent = t('Data preview');
      sub.textContent = dataSummary(cfg);
      status.className = run.error ? 'pill pill-warn' : 'pill';
      status.textContent = run.error ? t('Error') : t('Not trained yet');
      diff.innerHTML = '';
      return;
    }
    title.textContent = run.row ? `#${run.row.seq} · ${modelSummary(cfg)}` : modelSummary(cfg);
    sub.textContent = [dataSummary(cfg), lossSummary(cfg), optimSummary(cfg)].join('  |  ');

    const st = message ? ['warn', message]
      : run.live ? ['live', t('Training…')]
      : { timeout: ['warn', t('Stopped at the time limit')], diverged: ['warn', t('Diverged: the loss became infinite. Try a smaller learning rate.')] }[run.row?.status]
      || ['ok', t('Done')];
    status.className = `pill pill-${st[0]}`;
    status.textContent = st[1];

    const prev = run.row ? this.history.previous(this.history.rows.find(r => r.id === run.row.id)) : null;
    const d = run.row ? diffSummary(run.row, prev) : { changes: [] };
    if (!prev) { diff.innerHTML = ''; return; }
    const parts = d.changes.map(c => `<span class="chg"><b>${esc(c.label)}</b> ${esc(c.before)} → <em>${esc(c.now)}</em></span>`);
    const same = t('Same settings as #{n}', { n: d.prevSeq }), vs = t('vs #{n}:', { n: d.prevSeq }), lossLabel = t('Validation loss');
    if (!parts.length) parts.push(`<span class="muted">${same}</span>`);
    if (d.loss) {
      const better = d.loss.ratio < 0;
      parts.push(`<span class="loss-delta ${better ? 'better' : 'worse'}">${lossLabel} ${fmtLoss(d.loss.before)} → ${fmtLoss(d.loss.now)}
        (${better ? '▼' : '▲'} ${Math.abs(d.loss.ratio * 100).toFixed(0)}%)</span>`);
    }
    diff.innerHTML = `<span class="muted">${vs}</span> ${parts.join('')}`;
  }

  // ---------- MP4 ----------

  setupMp4Dialog() {
    const dlg = this.el.dialog;
    const form = dlg.querySelector('form');
    const bar = dlg.querySelector('[data-role="mp4-progress"] > div');
    form.addEventListener('submit', async e => {
      e.preventDefault();
      if (e.submitter?.value === 'cancel') { dlg.close(); return; }
      const run = this.run, f = form.elements;
      const save = form.querySelector('[value=save]');
      save.disabled = true;
      dlg.querySelector('[data-role="mp4-progress"]').hidden = false;
      try {
        const { blob, ext } = await exportVideo(run, {
          title: f.video_title.value.trim() || videoTitle(run.config),
          speed: +f.video_speed.value || 1,
          log: f.video_log.checked,
          view: this.player.view,
          onProgress: p => { bar.style.width = `${Math.round(p * 100)}%`; },
        });
        saveBlob(blob, `ml-playground-run${run.row?.seq ?? ''}.${ext}`);
        if (ext !== 'mp4') this.toast?.(t('This browser can only record WebM, so the video was saved as WebM.'), 'info');
        dlg.close();
      } catch (err) {
        this.toast?.(t('Could not make the video: {msg}', { msg: err.message }), 'error');
      } finally {
        save.disabled = false;
        dlg.querySelector('[data-role="mp4-progress"]').hidden = true;
        bar.style.width = '0';
      }
    });
  }

  openMp4Dialog() {
    if (!this.run?.meta) return;
    const f = this.el.dialog.querySelector('form').elements;
    f.video_title.value = videoTitle(this.run.config);
    f.video_speed.value = this.player.speed.value;
    this.el.dialog.showModal();
  }
}
