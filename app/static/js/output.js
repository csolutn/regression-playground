// Right-hand output: result header, animation player, MP4 export and history table.
// Used by the playground (students) and, read-only, by the teacher's student page.
import { t } from './i18n.js';
import { api } from './api.js';
import { Player } from './player.js';
import { History, localDate } from './history.js';
import { replayRun } from './train.js';
import { TRAINER_VERSION } from './nn.js';
import { exportVideo, saveBlob } from './recorder.js';
import { dataSummary, lossSummary, modelSummary, optimSummary, videoTitle } from './describe.js';

export class OutputPanel {
  constructor(root, { readonly = false, userId = null, guest = false, toast, onLoadSettings } = {}) {
    const $ = role => root.querySelector(`[data-role="${role}"]`);
    this.el = {
      title: $('result-title'), sub: $('result-sub'), status: $('result-status'),
      busy: $('viz-busy'), busyText: $('busy-text'), bar: root.querySelector('[data-role="progress"] > div'),
      mp4: $('mp4'), dialog: $('mp4-dialog'),
      count: $('history-count'), csv: $('csv'), more: $('history-more'),
    };
    this.login = root.dataset.login || '';
    this.userId = userId;
    this.guest = guest;                       // runs live only in this page (train.js makes their rows)
    this.toast = toast;
    this.cache = new Map();                   // run id → full run (frames etc.), for the rest of the visit
    this.replaying = null;                    // the AbortController of the run being trained again (view)
    this.player = new Player(root);
    this.history = new History($('history'), {
      readonly,
      popup: $('filter-pop'),
      onView: row => this.view(row),
      onLoad: async row => onLoadSettings?.(await this.fetchSettings(row)),
      onDelete: row => this.remove(row),
      onNeedAll: () => this.loadOlder(true),
    });
    this.loading = null;                      // the running loadOlder()
    this.el.mp4.addEventListener('click', () => this.openMp4Dialog());
    this.setupMp4Dialog();
    this.setupHistoryTools();
    this.showHeader(null);
  }

  get run() { return this.player.run; }

  async init() {
    if (this.guest) return;
    try {
      const page = await api.runs(this.userId);
      this.history.setRows(page.runs, page);
      if (page.runs[0]) await this.view(page.runs[0]);
    } catch (err) {
      this.toast?.(err.message, 'error');
    }
  }

  // the next page of older runs, or all of them (the filters and the CSV need every run)
  async loadOlder(all = false) {
    while (this.loading) await this.loading;
    const { older, rows } = this.history;
    if (!older) return;
    this.loading = (async () => {
      this.el.more.disabled = true;
      try {
        const page = await api.runs(this.userId, { before: (rows.at(-1)?.id ?? older.id + 1), all });
        this.history.addOlder(page.runs, page.older, page.total);
      } catch (err) {
        this.toast?.(err.message, 'error');
      } finally {
        this.el.more.disabled = false;
        this.loading = null;
      }
    })();
    await this.loading;
  }

  // A run with all it needs to be shown: a server run as stored, a browser run trained again from its
  // settings (only its results are stored), with a progress bar over the plot meanwhile
  async fetchRun(row, signal) {
    if (!this.cache.has(row.id)) {
      const d = await api.run(row.id);
      this.cache.set(row.id, d.replay ? await this.replay(d, signal)
        : { row: d.run, config: d.config, meta: d.meta, frames: d.frames, loss: d.loss, params: d.params });
    }
    return this.cache.get(row.id);
  }

  // just its settings and row, to load them: no need to train it again
  async fetchSettings(row) {
    if (this.cache.has(row.id)) return this.cache.get(row.id);
    const d = await api.run(row.id);
    return { row: d.run, config: d.config };
  }

  // A browser run trained again with the same settings and seed on the same data, which makes the same
  // numbers in the same browser. note: why its replay may differ from what it showed when it was trained.
  async replay(d, signal) {
    const meta = await api.preview(d.config), label = t('Rebuilding the training…');
    const progress = p => this.setProgress(p, `${label} ${Math.round(p * 100)}%`);
    progress(0);
    const made = await replayRun(d.config, meta, d.run.steps, progress, signal);
    const saved = d.run.final_val, now = made.end.final_val;
    const differs = saved != null && now != null && Math.abs(now - saved) > 1e-3 * Math.abs(saved);
    const note = d.replay.trainer !== TRAINER_VERSION || d.replay.data_changed
      ? t('The training code has changed since this run: its replay may differ from what it showed then.')
      : differs ? t('Trained again in this browser: the numbers differ a little from the saved ones.') : null;
    return { row: d.run, config: d.config, meta, frames: made.frames, loss: made.loss, params: made.params, note };
  }

  async view(row) {
    if (this.run?.live) { this.toast?.(t('Wait until training finishes.'), 'info'); return; }
    this.replaying?.abort();                  // another row was clicked while one was being trained again
    const ac = this.replaying = new AbortController();
    try {
      const run = await this.fetchRun(row, ac.signal);
      this.player.setRun(run);
      this.history.setActive(row.id);
      this.showHeader(run);
      if (run.note && !run.noted) { run.noted = true; this.toast?.(run.note, 'info'); }
    } catch (err) {
      if (err.name !== 'AbortError') this.toast?.(err.message, 'error');
    } finally {
      if (this.replaying === ac) { this.replaying = null; this.setProgress(null); }
    }
  }

  async remove(row) {
    if (!confirm(t('Delete run #{n}? This cannot be undone.', { n: row.seq }))) return;
    try {
      if (!this.guest) await api.deleteRun(row.id);
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
    const { count, csv, more } = this.el;
    this.history.onRender = (shown, total) => {
      count.textContent = this.history.filtered ? t('{n} of {total} runs', { n: shown, total }) : '';
      csv.disabled = shown === 0;
      more.hidden = !this.history.older;
      const loaded = { n: this.history.rows.length, total: this.history.total };
      more.textContent = t('Load more ({n} of {total})', loaded);
    };
    more.addEventListener('click', () => this.loadOlder());
    csv.addEventListener('click', async () => {
      await this.loadOlder(true);
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
    this.replaying?.abort();
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
    } else if (ev.type === 'params') {
      run.params = { init: ev.init, final: ev.final, contrib: ev.contrib, path_epochs: ev.path_epochs, paths: ev.paths };
    } else if (ev.type === 'end') {          // done: shown at once, while it is saved ('saved' brings its row)
      run.end = ev;
      run.live = false;
      this.player.endLive();
      this.setProgress(null);
      this.showHeader(run);
      this.player.playFromStart();          // replay it from epoch 0 at the chosen speed
    } else if (ev.type === 'error') {
      if (!run.end) this.stopLive(run, t('Error'));
      else if (this.run === run) this.showHeader(run, t('Not saved'));   // trained, but saving it failed
      this.toast?.(ev.message, 'error');
    } else if (ev.type === 'saved') {
      run.row = ev.run;
      this.cache.set(ev.run.id, run);
      this.history.add(ev.run);
      if (this.run === run) {               // still shown (no other run started meanwhile)
        this.history.setActive(ev.run.id);
        this.showHeader(run);
      }
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

  // ---------- header: what this run is ----------

  showHeader(run, message) {
    const { title, sub, status, mp4 } = this.el;
    mp4.disabled = !run?.meta || run.live || run.frames.length < 2;
    if (!run) {
      title.textContent = t('Result');
      sub.textContent = t('Set up the experiment on the left and press Train.');
      status.textContent = ''; status.className = '';
      return;
    }
    const cfg = run.config;
    if (run.preview) {
      title.textContent = t('Data preview');
      sub.textContent = dataSummary(cfg);
      status.className = run.error ? 'pill pill-warn' : 'pill';
      status.textContent = run.error ? t('Error') : t('Not trained yet');
      return;
    }
    title.textContent = run.row ? `#${run.row.seq} · ${modelSummary(cfg)}` : modelSummary(cfg);
    sub.textContent = [dataSummary(cfg), lossSummary(cfg), optimSummary(cfg)].join('  |  ');

    const st = message ? ['warn', message]
      : run.live ? ['live', t('Training…')]
      : { timeout: ['warn', t('Stopped at the time limit')], diverged: ['warn', t('Diverged: the loss became infinite. Try a smaller learning rate.')] }[(run.row || run.end)?.status]
      || ['ok', t('Done')];
    status.className = `pill pill-${st[0]}`;
    status.textContent = st[1];
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
