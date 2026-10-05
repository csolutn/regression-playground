// Animation of a run in the browser: play / pause / seek with the epoch slider.
// The server only sends numbers (frames); every image is drawn here.
// While a run is training only its data is drawn (the output panel shows a progress bar over it):
// frames arrive much faster than they are played back, which looks like instant convergence.
import { DEFAULT_VIEW, drawLandscape, drawLoss, drawPrediction, drawSlices, fitCanvas, fmtLoss, screenTheme } from './plot.js';
import { stepLabel } from './describe.js';
import { hasLandscape, hasSlices, landscape, paramSlices } from './landscape.js';

const RUN_MS = 16_000;               // a whole run takes about this long at 1×
const MAX_FRAME_MS = 267;           // runs with few frames (e.g. a shallow tree) don't flash by
const END_HOLD_MS = 1200;           // pause on the last frame before looping

// time per frame for a run of n frames at the given speed
export function frameMs(n, speed = 1) {
  return Math.min(MAX_FRAME_MS, RUN_MS / Math.max(1, n)) / speed;
}

export class Player {
  constructor(root) {
    const $ = role => root.querySelector(`[data-role="${role}"]`);
    this.canvas = $('pred-canvas');
    this.empty = $('viz-empty');
    this.emptyText = this.empty.textContent;
    this.slider = $('frame-slider');
    this.playBtn = $('play');
    this.prevBtn = $('prev');
    this.nextBtn = $('next');
    this.speed = $('speed');
    this.trainText = $('train-text');
    this.valText = $('val-text');
    this.loss = { open: $('loss-open'), dialog: $('loss-dialog'), canvas: $('loss-canvas'), log: $('loss-log') };
    this.land = { open: $('land-open'), dialog: $('land-dialog'), canvas: $('land-canvas'), log: $('land-log'),
                  hintSurface: $('land-hint-surface'), hintSlices: $('land-hint-slices'),
                  play: $('land-play'), slider: $('land-slider'), step: $('land-step'),
                  cut: $('land-cut'), cutCheck: $('land-cut-check'), cutKey: $('land-cut-key'), cutHint: $('land-cut-hint') };
    this.landView = { ...DEFAULT_VIEW };
    this.run = null;
    this.idx = 0;
    this.follow = false;          // live training: keep showing the newest frame
    this.timer = null;
    this.view = { ...DEFAULT_VIEW };

    for (const btn of [this.playBtn, this.land.play]) btn.addEventListener('click', () => (this.timer ? this.pause() : this.play()));
    this.prevBtn.addEventListener('click', () => this.seek(this.idx - 1));
    this.nextBtn.addEventListener('click', () => this.seek(this.idx + 1));
    for (const sl of [this.slider, this.land.slider]) sl.addEventListener('input', () => this.seek(+sl.value));
    document.addEventListener('keydown', e => this.onKey(e));
    this.enablePopup(this.loss, $('loss-close'));
    this.enablePopup(this.land, $('land-close'));
    this.land.cut.addEventListener('change', () => { this.showCutKey(); this.render(); });
    for (const c of [this.canvas, this.loss.canvas, this.land.canvas]) new ResizeObserver(() => this.render()).observe(c);
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.render());
    this.enableRotate(this.canvas, this.view, () => this.run?.meta?.n_inputs === 2);
    this.enableRotate(this.land.canvas, this.landView, () => this.landKind === 'surface');
    this.sync();
  }

  get frames() { return this.run?.frames || []; }
  get speedFactor() { return +this.speed.value || 1; }
  get frameMs() { return frameMs(this.frames.length, this.speedFactor); }

  setRun(run, { live = false } = {}) {
    this.pause();
    this.run = run;
    this.follow = live;
    this.idx = Math.max(0, this.frames.length - 1);
    this.sync();
  }

  // new frames arrived while training
  update() {
    if (this.follow) this.idx = Math.max(0, this.frames.length - 1);
    this.sync();
  }

  endLive() {
    this.follow = false;
    this.sync();
  }

  // jump to frame i (pauses playback)
  seek(i) {
    clearTimeout(this.timer);
    this.timer = null;
    const last = this.frames.length - 1;
    this.idx = Math.max(0, Math.min(last, i));
    this.follow = this.run?.live && this.idx === last;
    this.sync();
  }

  play() {
    if (!this.frames.length) return;
    this.follow = false;
    if (this.idx >= this.frames.length - 1) this.idx = 0;
    const tick = () => {
      const atEnd = this.idx >= this.frames.length - 1;
      if (atEnd && this.run?.live) { this.follow = true; this.pause(); return; }
      this.idx = atEnd ? 0 : this.idx + 1;
      this.sync();
      const last = this.idx >= this.frames.length - 1;
      this.timer = setTimeout(tick, this.frameMs + (last ? END_HOLD_MS : 0));
    };
    this.timer = setTimeout(tick, this.frameMs);
    this.sync();
  }

  playFromStart() {
    this.pause();
    this.idx = 0;
    this.play();
  }

  pause() {
    clearTimeout(this.timer);
    this.timer = null;
    this.sync();
  }

  onKey(e) {
    const dialog = e.target.closest('dialog');     // the loss graph and landscape follow the frame, so they may play
    if (dialog && dialog !== this.loss.dialog && dialog !== this.land.dialog) return;
    if (!this.frames.length || this.run?.live || e.target.closest('input, select, textarea, button') || e.metaKey || e.ctrlKey) return;
    if (e.key === ' ') { e.preventDefault(); this.timer ? this.pause() : this.play(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); this.seek(this.idx - 1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); this.seek(this.idx + 1); }
  }

  // controls and numbers for the current frame
  sync() {
    const n = this.run?.live ? 0 : this.frames.length, f = n ? this.frames[this.idx] : null;
    for (const sl of [this.slider, this.land.slider]) { sl.max = Math.max(0, n - 1); sl.value = this.idx; }
    for (const el of [this.slider, this.land.slider, this.playBtn, this.land.play, this.prevBtn, this.nextBtn]) el.disabled = n < 2;
    this.loss.open.disabled = !this.run?.meta || !!this.run.preview || !!this.run.live;
    const kind = this.landKind;
    this.land.open.hidden = !kind;
    if (!kind && this.land.dialog.open) this.land.dialog.close();
    this.land.canvas.classList.toggle('is-slices', kind === 'slices');
    this.land.hintSurface.hidden = kind !== 'surface';
    this.land.hintSlices.hidden = this.land.cutCheck.hidden = kind !== 'slices';
    for (const btn of [this.playBtn, this.land.play]) {
      btn.classList.toggle('is-playing', !!this.timer);
      btn.setAttribute('aria-pressed', String(!!this.timer));
    }
    this.empty.hidden = !!this.run?.meta;
    this.empty.textContent = this.run?.error || this.emptyText;
    this.empty.classList.toggle('is-error', !!this.run?.error);
    if (f && this.run?.meta) {
      this.land.step.textContent = `${stepLabel(this.run.meta.model)} ${f.step.toLocaleString()} / ${this.run.meta.total_steps.toLocaleString()}`;
      this.trainText.textContent = fmtLoss(f.train);
      this.valText.textContent = fmtLoss(f.val);
    } else {
      this.land.step.textContent = '—';
      this.trainText.textContent = this.valText.textContent = '—';
    }
    this.render();
  }

  render() {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = null;
      const { ctx, w, h } = fitCanvas(this.canvas);
      const th = screenTheme();
      ctx.fillStyle = th.bg;
      ctx.fillRect(0, 0, w, h);
      if (!this.run?.meta) return;
      const s = Math.max(0.8, Math.min(1.25, w / 700));
      const preview = !!(this.run.preview || this.run.live);     // data before / while training: points only
      drawPrediction(ctx, { x: 0, y: 0, w, h }, this.run, this.idx, th, s, { view: this.view, dataOnly: preview, badges: !preview });
      if (this.loss.dialog.open) this.renderLoss(th);
      if (this.land.dialog.open) this.renderLandscape(th);
    });
  }

  // the loss landscape of the run shown, computed when first opened: the whole surface of one-input linear
  // regression ('surface'), slices along the weights that mattered most for other runs that saved weights
  get landKind() {
    if (!this.run || this.run.preview || this.run.live) return null;
    return hasLandscape(this.run) ? 'surface' : hasSlices(this.run) ? 'slices' : null;
  }

  showCutKey() { this.land.cutKey.hidden = this.land.cutHint.hidden = !this.land.cut.checked; }

  renderLandscape(th) {
    const { ctx, w, h } = fitCanvas(this.land.canvas);
    ctx.fillStyle = th.bg;
    ctx.fillRect(0, 0, w, h);
    const kind = this.landKind, s = Math.max(0.8, Math.min(1.1, w / 700));
    const opts = { view: this.landView, log: this.land.log.checked, cut: this.land.cut.checked };
    if (kind === 'surface') {
      this.run.landscape ||= landscape(this.run);
      if (this.run.landscape) drawLandscape(ctx, { x: 0, y: 0, w, h }, this.run.landscape, this.idx, this.run, th, s, opts);
    } else if (kind === 'slices') {
      this.run.slices ||= paramSlices(this.run);
      drawSlices(ctx, { x: 0, y: 0, w, h }, this.run.slices, this.idx, this.run, th, s, opts);
    }
  }

  // loss curve up to the current frame, so it moves with the animation
  renderLoss(th) {
    const { ctx, w, h } = fitCanvas(this.loss.canvas);
    ctx.fillStyle = th.bg;
    ctx.fillRect(0, 0, w, h);
    const f = this.frames[this.idx];
    if (f) drawLoss(ctx, { x: 0, y: 0, w, h }, this.run, f.step, th, Math.max(0.8, Math.min(1.1, w / 700)), { log: this.loss.log.checked });
  }

  // "Show loss graph" / "Show loss landscape" open a popup; ✕, Esc, a click outside or the link again closes it
  enablePopup({ open, dialog, log }, closeBtn) {
    const close = () => dialog.close();
    open.addEventListener('click', () => (dialog.open ? close() : (dialog.showModal(), this.render())));
    closeBtn.addEventListener('click', close);
    dialog.addEventListener('click', e => { if (e.target === dialog) close(); });
    log.addEventListener('change', () => this.render());
  }

  // drag to rotate a 3D plot (the 2-input surface, the loss landscape)
  enableRotate(canvas, view, enabled) {
    let start = null;
    canvas.addEventListener('pointerdown', e => {
      if (!enabled()) return;
      start = { x: e.clientX, y: e.clientY, ...view };
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', e => {
      if (!start) return;
      view.az = start.az + (e.clientX - start.x) * 0.01;
      view.el = Math.max(0.05, Math.min(1.45, start.el + (e.clientY - start.y) * 0.01));
      this.render();
    });
    const end = () => { start = null; };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
  }
}
