// Animation of a run in the browser: play / pause / seek with the epoch slider.
// The server only sends numbers (frames); every image is drawn here.
import { DEFAULT_VIEW, drawPrediction, fitCanvas, fmtLoss, screenTheme } from './plot.js';
import { stepLabel } from './describe.js';

export const BASE_FRAME_MS = 200;   // time per frame at 1× (the notebook GIF used 200 ms)
const END_HOLD_MS = 1200;           // pause on the last frame before looping

export class Player {
  constructor(root) {
    const $ = role => root.querySelector(`[data-role="${role}"]`);
    this.canvas = $('pred-canvas');
    this.empty = $('viz-empty');
    this.slider = $('frame-slider');
    this.playBtn = $('play');
    this.prevBtn = $('prev');
    this.nextBtn = $('next');
    this.speed = $('speed');
    this.stepText = $('step-text');
    this.trainText = $('train-text');
    this.valText = $('val-text');
    this.run = null;
    this.idx = 0;
    this.follow = false;          // live training: keep showing the newest frame
    this.timer = null;
    this.view = { ...DEFAULT_VIEW };

    this.playBtn.addEventListener('click', () => (this.timer ? this.pause() : this.play()));
    this.prevBtn.addEventListener('click', () => this.seek(this.idx - 1));
    this.nextBtn.addEventListener('click', () => this.seek(this.idx + 1));
    this.slider.addEventListener('input', () => this.seek(+this.slider.value));
    document.addEventListener('keydown', e => this.onKey(e));
    new ResizeObserver(() => this.render()).observe(this.canvas);
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.render());
    this.enableRotate();
    this.sync();
  }

  get frames() { return this.run?.frames || []; }
  get speedFactor() { return +this.speed.value || 1; }

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
      this.timer = setTimeout(tick, BASE_FRAME_MS / this.speedFactor + (last ? END_HOLD_MS : 0));
    };
    this.timer = setTimeout(tick, BASE_FRAME_MS / this.speedFactor);
    this.sync();
  }

  pause() {
    clearTimeout(this.timer);
    this.timer = null;
    this.sync();
  }

  onKey(e) {
    if (!this.frames.length || e.target.closest('input, select, textarea, button, dialog') || e.metaKey || e.ctrlKey) return;
    if (e.key === ' ') { e.preventDefault(); this.timer ? this.pause() : this.play(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); this.seek(this.idx - 1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); this.seek(this.idx + 1); }
  }

  // controls and numbers for the current frame
  sync() {
    const n = this.frames.length, f = this.frames[this.idx];
    this.slider.max = Math.max(0, n - 1);
    this.slider.value = this.idx;
    for (const el of [this.slider, this.playBtn, this.prevBtn, this.nextBtn]) el.disabled = n < 2;
    this.playBtn.classList.toggle('is-playing', !!this.timer);
    this.playBtn.setAttribute('aria-pressed', String(!!this.timer));
    this.empty.hidden = !!this.run?.meta;
    if (f && this.run?.meta) {
      this.stepText.textContent = `${stepLabel(this.run.meta.model)} ${f.step.toLocaleString()} / ${this.run.meta.total_steps.toLocaleString()}`;
      this.trainText.textContent = fmtLoss(f.train);
      this.valText.textContent = fmtLoss(f.val);
    } else {
      this.stepText.textContent = '—';
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
      drawPrediction(ctx, { x: 0, y: 0, w, h }, this.run, this.idx, th, s, { view: this.view });
    });
  }

  // drag to rotate the 3D surface (2 input variables)
  enableRotate() {
    let start = null;
    this.canvas.addEventListener('pointerdown', e => {
      if (this.run?.meta?.n_inputs !== 2) return;
      start = { x: e.clientX, y: e.clientY, ...this.view };
      this.canvas.setPointerCapture(e.pointerId);
    });
    this.canvas.addEventListener('pointermove', e => {
      if (!start) return;
      this.view.az = start.az + (e.clientX - start.x) * 0.01;
      this.view.el = Math.max(0.05, Math.min(1.45, start.el + (e.clientY - start.y) * 0.01));
      this.render();
    });
    const end = () => { start = null; };
    this.canvas.addEventListener('pointerup', end);
    this.canvas.addEventListener('pointercancel', end);
  }
}
