// Canvas drawing shared by the screen and the MP4 export.
// Every function draws into a rectangle {x, y, w, h} of a 2D context; `s` scales fonts and lines.
import { t } from './i18n.js';
import { configBadge, stepLabel } from './describe.js';

export const FONT = 'system-ui, -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", "Segoe UI", sans-serif';
export const MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace';

const SERIES = {
  train: '#e4574e', val: '#f2a33a', pred: '#3b7ddd', truth: '#6b7280',
};
export const LIGHT = {
  ...SERIES, bg: '#ffffff', panel: '#fbfcfd', grid: '#e3e8ef', frame: '#cfd6df',
  text: '#111827', muted: '#5b6472', badge: 'rgba(255,255,255,.92)',
};

export function screenTheme() {
  const cs = getComputedStyle(document.documentElement);
  const v = (name, fallback) => cs.getPropertyValue(name).trim() || fallback;
  return {
    ...SERIES, bg: v('--surface', LIGHT.bg), panel: v('--plot-panel', LIGHT.panel), grid: v('--plot-grid', LIGHT.grid),
    frame: v('--plot-frame', LIGHT.frame), text: v('--text', LIGHT.text), muted: v('--muted', LIGHT.muted),
    badge: v('--plot-badge', LIGHT.badge),
  };
}

// ---------- numbers ----------

export function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const r = raw / mag;
  const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export function fmt(v) {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e5 || a < 1e-3) return v.toExponential(0);
  return String(+v.toPrecision(4));
}

export function fmtLoss(v) {
  if (v == null || !Number.isFinite(v)) return '—';
  return Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : v.toPrecision(3);
}

function extent(arrays) {
  let lo = Infinity, hi = -Infinity;
  for (const a of arrays) for (const v of a || []) {
    if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  }
  if (!Number.isFinite(lo)) return [0, 1];
  return [lo, hi];
}

// y range fixed for the whole run: data and true function, plus 10 % margin
export function yRange(meta) {
  const [lo, hi] = extent([meta.y_train, meta.y_val, meta.y_true]);
  const m = Math.max(0.1 * (hi - lo), 1e-3);
  return [lo - m, hi + m];
}

// ---------- building blocks ----------

function font(px, weight = 400, family = FONT) {
  return `${weight} ${px}px ${family}`;
}

function panel(ctx, P, th, s) {
  ctx.fillStyle = th.panel;
  ctx.fillRect(P.x, P.y, P.w, P.h);
  ctx.strokeStyle = th.frame;
  ctx.lineWidth = 1.5 * s;
  ctx.strokeRect(P.x, P.y, P.w, P.h);
}

function grid(ctx, P, th, s, xTicks, yTicks, X, Y, xLabel = fmt, yLabel = fmt) {
  ctx.strokeStyle = th.grid;
  ctx.lineWidth = 1 * s;
  ctx.beginPath();
  for (const v of xTicks) { const x = X(v); ctx.moveTo(x, P.y); ctx.lineTo(x, P.y + P.h); }
  for (const v of yTicks) { const y = Y(v); ctx.moveTo(P.x, y); ctx.lineTo(P.x + P.w, y); }
  ctx.stroke();
  ctx.fillStyle = th.muted;
  ctx.font = font(11 * s);
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const v of xTicks) ctx.fillText(xLabel(v), X(v), P.y + P.h + 5 * s);
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const v of yTicks) ctx.fillText(yLabel(v), P.x - 6 * s, Y(v));
}

function badge(ctx, lines, x, y, align, th, s, family = FONT, size = 12) {
  ctx.font = font(size * s, 400, family);
  const lh = size * 1.35 * s, padX = 8 * s, padY = 5 * s;
  const w = Math.max(...lines.map(l => ctx.measureText(l).width)) + 2 * padX;
  const h = lines.length * lh + 2 * padY;
  const bx = align === 'right' ? x - w : x;
  ctx.fillStyle = th.badge;
  ctx.strokeStyle = th.frame;
  ctx.lineWidth = 1.2 * s;
  roundRect(ctx, bx, y, w, h, 5 * s);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = th.text;
  ctx.textBaseline = 'top';
  ctx.textAlign = align === 'right' ? 'right' : 'left';
  lines.forEach((l, i) => ctx.fillText(l, align === 'right' ? bx + w - padX : bx + padX, y + padY + i * lh));
  return h;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function legend(ctx, items, x, y, align, th, s, inside = true) {
  ctx.font = font(12 * s);
  const sw = 20 * s, gap = 6 * s, lh = 18 * s, pad = 7 * s;
  const w = Math.max(...items.map(i => ctx.measureText(i.label).width)) + sw + gap + 2 * pad;
  const h = items.length * lh + 2 * pad - 4 * s;
  const bx = align === 'right' ? x - w : x, by = y - (inside ? h : 0);
  ctx.fillStyle = th.badge; ctx.strokeStyle = th.frame; ctx.lineWidth = 1.2 * s;
  roundRect(ctx, bx, by, w, h, 5 * s); ctx.fill(); ctx.stroke();
  items.forEach((it, i) => {
    const cy = by + pad + i * lh + 5 * s;
    ctx.strokeStyle = ctx.fillStyle = it.color;
    if (it.kind === 'dot') {
      ctx.beginPath(); ctx.arc(bx + pad + sw / 2, cy, 3.5 * s, 0, 7); ctx.fill();
    } else {
      ctx.lineWidth = (it.width || 3) * s;
      ctx.setLineDash(it.dash ? it.dash.map(d => d * s) : []);
      ctx.beginPath(); ctx.moveTo(bx + pad, cy); ctx.lineTo(bx + pad + sw, cy); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.fillStyle = th.text; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(it.label, bx + pad + sw + gap, cy);
  });
}

// Legend laid out in one row, centered at x
function legendRow(ctx, items, cx, y, th, s) {
  ctx.font = font(12 * s);
  const sw = 22 * s, gap = 6 * s, space = 18 * s;
  const widths = items.map(i => sw + gap + ctx.measureText(i.label).width);
  let x = cx - (widths.reduce((a, b) => a + b, 0) + space * (items.length - 1)) / 2;
  items.forEach((it, i) => {
    ctx.strokeStyle = it.color; ctx.lineWidth = (it.width || 3) * s;
    ctx.setLineDash(it.dash ? it.dash.map(d => d * s) : []);
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + sw, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = th.text; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(it.label, x + sw + gap, y);
    x += widths[i] + space;
  });
}

function polyline(ctx, xs, ys, X, Y) {
  ctx.beginPath();
  let pen = false;
  for (let i = 0; i < xs.length; i++) {
    const v = ys[i];
    if (v == null || !Number.isFinite(v)) { pen = false; continue; }
    const px = X(xs[i]), py = Y(v);
    if (pen) ctx.lineTo(px, py); else { ctx.moveTo(px, py); pen = true; }
  }
  ctx.stroke();
}

// ---------- ② prediction ----------

// Frame i of a run: the model's prediction plus Epoch and settings badges
// opts.dataOnly: just the data points (before training), without the true function or a prediction
export function drawPrediction(ctx, r, run, idx, th, s = 1, opts = {}) {
  const frame = opts.dataOnly ? null : run.frames[idx];
  const m = opts.dataOnly ? { ...run.meta, y_true: null } : run.meta;
  const P = m.n_inputs === 1
    ? drawPrediction1D(ctx, r, m, frame?.pred, th, s, !opts.dataOnly)
    : drawPrediction2D(ctx, r, m, frame?.pred, th, s, opts.view || DEFAULT_VIEW, !opts.dataOnly);
  if (opts.badges === false) return;
  const total = run.meta.total_steps;
  const step = frame ? frame.step : 0;
  badge(ctx, [`${stepLabel(run.meta.model)} ${step.toLocaleString()}/${total.toLocaleString()}`],
    P.x + 10 * s, P.y + 10 * s, 'left', th, s, FONT, 14);
  badge(ctx, configBadge(run.config), P.x + P.w - 10 * s, P.y + 10 * s, 'right', th, s, MONO, 11.5);
}

function drawPrediction1D(ctx, r, m, pred, th, s, withPred) {
  const xs = m.axes[0];
  const [ylo, yhi] = yRange(m);
  const xlo = xs[0], xhi = xs[xs.length - 1];
  const P = { x: r.x + 54 * s, y: r.y + 6 * s, w: r.w - 64 * s, h: r.h - 46 * s };
  const X = v => P.x + (v - xlo) / (xhi - xlo || 1) * P.w;
  const Y = v => P.y + P.h - (v - ylo) / (yhi - ylo) * P.h;

  panel(ctx, P, th, s);
  grid(ctx, P, th, s, niceTicks(xlo, xhi, 6), niceTicks(ylo, yhi, 5), X, Y);
  ctx.fillStyle = th.text; ctx.font = font(13 * s);
  ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
  ctx.fillText(m.feature_names[0], P.x + P.w / 2, r.y + r.h);
  ctx.save();
  ctx.translate(r.x + 12 * s, P.y + P.h / 2); ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = 'middle'; ctx.fillText(m.target_name, 0, 0);
  ctx.restore();

  ctx.save();
  ctx.beginPath(); ctx.rect(P.x, P.y, P.w, P.h); ctx.clip();
  const dots = (x, y, color, alpha, rad) => {
    ctx.globalAlpha = alpha; ctx.fillStyle = color;
    for (let i = 0; i < y.length; i++) { ctx.beginPath(); ctx.arc(X(x[i]), Y(y[i]), rad * s, 0, 7); ctx.fill(); }
    ctx.globalAlpha = 1;
  };
  dots(m.x_train[0], m.y_train, th.train, 0.6, 2.8);
  dots(m.x_val[0], m.y_val, th.val, 0.75, 2.8);
  if (m.y_true) {
    ctx.strokeStyle = th.truth; ctx.lineWidth = 1.6 * s; ctx.setLineDash([6 * s, 5 * s]);
    polyline(ctx, xs, m.y_true, X, Y); ctx.setLineDash([]);
  }
  if (pred) {
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.strokeStyle = th.pred; ctx.globalAlpha = 0.18; ctx.lineWidth = 10 * s;
    polyline(ctx, xs, pred, X, Y);
    ctx.globalAlpha = 1; ctx.lineWidth = 3.4 * s;
    polyline(ctx, xs, pred, X, Y);
  }
  ctx.restore();

  const items = [
    { kind: 'dot', color: th.train, label: t('Training data') },
    { kind: 'dot', color: th.val, label: t('Validation data') },
  ];
  if (m.y_true) items.push({ color: th.truth, width: 1.6, dash: [6, 5], label: t('True function') });
  if (withPred) items.push({ color: th.pred, width: 3.4, label: t('Prediction') });
  legend(ctx, items, P.x + P.w - 10 * s, P.y + P.h - 10 * s, 'right', th, s);
  return P;
}

// ---------- 2 inputs: rotatable 3D surface ----------

export const DEFAULT_VIEW = { az: -0.65, el: 0.5 };

function coolwarm(u) {
  const stops = [[59, 76, 192], [124, 159, 249], [221, 221, 221], [245, 148, 114], [180, 4, 38]];
  const x = Math.min(0.9999, Math.max(0, u)) * (stops.length - 1);
  const i = Math.floor(x), f = x - i, a = stops[i], b = stops[i + 1];
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(',')})`;
}

function drawPrediction2D(ctx, r, m, pred, th, s, view, withPred) {
  const [ax1, ax2] = m.axes, n1 = ax1.length, n2 = ax2.length;
  const [zlo, zhi] = yRange(m);
  const P = { x: r.x + 4 * s, y: r.y + 4 * s, w: r.w - 8 * s, h: r.h - 8 * s };
  panel(ctx, P, th, s);

  const u = v => (v - ax1[0]) / (ax1[n1 - 1] - ax1[0] || 1) * 2 - 1;
  const v_ = v => (v - ax2[0]) / (ax2[n2 - 1] - ax2[0] || 1) * 2 - 1;
  const w = z => Math.max(-1, Math.min(1, (z - zlo) / (zhi - zlo) * 2 - 1)) * 0.72;
  const ca = Math.cos(view.az), sa = Math.sin(view.az), ce = Math.cos(view.el), se = Math.sin(view.el);
  const scale = Math.min(P.w / 3.4, P.h / 3.0);
  const cx = P.x + P.w / 2, cy = P.y + P.h * 0.5;
  const project = (a, b, c) => {
    const xr = a * ca - b * sa, yr = a * sa + b * ca;
    return [cx + xr * scale, cy - (yr * se + c * ce) * scale, -yr * ce + c * se];
  };

  // floor and back walls of the box
  ctx.strokeStyle = th.frame; ctx.lineWidth = 1.2 * s;
  const floor = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => project(a, b, -0.72));
  ctx.beginPath(); floor.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); ctx.stroke();
  const back = floor.reduce((best, p, i) => (p[2] < floor[best][2] ? i : best), 0);
  const [bx, by] = floor[back], [tx, ty] = project(...[[-1, -1], [1, -1], [1, 1], [-1, 1]][back], 0.72);
  ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(tx, ty); ctx.stroke();

  const items = [];
  if (pred) {
    const z = (i, j) => pred[j * n1 + i];
    for (let j = 0; j < n2 - 1; j++) for (let i = 0; i < n1 - 1; i++) {
      const zs = [z(i, j), z(i + 1, j), z(i + 1, j + 1), z(i, j + 1)];
      const pts = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]].map(([a, b], k) => project(u(ax1[a]), v_(ax2[b]), w(zs[k])));
      const zm = (zs[0] + zs[1] + zs[2] + zs[3]) / 4;
      items.push({ d: (pts[0][2] + pts[1][2] + pts[2][2] + pts[3][2]) / 4, kind: 'quad', pts, color: coolwarm((zm - zlo) / (zhi - zlo)) });
    }
  }
  if (m.y_true) {                       // true function as a light wireframe
    const zt = (i, j) => m.y_true[j * n1 + i];
    const stride = Math.max(1, Math.round(n1 / 10));
    for (let j = 0; j < n2; j += stride) for (let i = 0; i < n1 - 1; i++) {
      const a = project(u(ax1[i]), v_(ax2[j]), w(zt(i, j))), b = project(u(ax1[i + 1]), v_(ax2[j]), w(zt(i + 1, j)));
      items.push({ d: (a[2] + b[2]) / 2 + 1e-3, kind: 'seg', a, b });
    }
    for (let i = 0; i < n1; i += stride) for (let j = 0; j < n2 - 1; j++) {
      const a = project(u(ax1[i]), v_(ax2[j]), w(zt(i, j))), b = project(u(ax1[i]), v_(ax2[j + 1]), w(zt(i, j + 1)));
      items.push({ d: (a[2] + b[2]) / 2 + 1e-3, kind: 'seg', a, b });
    }
  }
  const [x1s, x2s] = m.x_train;
  for (let k = 0; k < m.y_train.length; k++) {
    const p = project(u(x1s[k]), v_(x2s[k]), w(m.y_train[k]));
    items.push({ d: p[2] + 2e-3, kind: 'dot', p });
  }
  items.sort((a, b) => a.d - b.d);      // painter's algorithm: far to near

  ctx.save();
  ctx.beginPath(); ctx.rect(P.x, P.y, P.w, P.h); ctx.clip();
  for (const it of items) {
    if (it.kind === 'quad') {
      ctx.fillStyle = ctx.strokeStyle = it.color; ctx.lineWidth = 0.6 * s;
      ctx.beginPath(); it.pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath();
      ctx.globalAlpha = 0.92; ctx.fill(); ctx.stroke(); ctx.globalAlpha = 1;
    } else if (it.kind === 'seg') {
      ctx.strokeStyle = th.text; ctx.globalAlpha = 0.28; ctx.lineWidth = 0.9 * s;
      ctx.beginPath(); ctx.moveTo(it.a[0], it.a[1]); ctx.lineTo(it.b[0], it.b[1]); ctx.stroke(); ctx.globalAlpha = 1;
    } else {
      ctx.fillStyle = th.train; ctx.globalAlpha = 0.75;
      ctx.beginPath(); ctx.arc(it.p[0], it.p[1], 2.2 * s, 0, 7); ctx.fill(); ctx.globalAlpha = 1;
    }
  }
  ctx.restore();

  // axis names at the middle of the two front floor edges
  ctx.fillStyle = th.muted; ctx.font = font(13 * s); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  const lab = (a, b, text) => { const [x, y] = project(a, b, -0.72); ctx.fillText(text, x, y + 14 * s); };
  lab(0, ca > 0 ? -1.12 : 1.12, m.feature_names[0]);
  lab(sa > 0 ? 1.12 : -1.12, 0, m.feature_names[1]);

  const note = [{ kind: 'dot', color: th.train, label: t('Training data') }];
  if (withPred) note.push({ color: coolwarm(0.8), width: 6, label: t('Prediction (surface)') });
  if (m.y_true) note.push({ color: th.muted, width: 1, label: t('True function (grid)') });
  legend(ctx, note, P.x + P.w - 10 * s, P.y + P.h - 10 * s, 'right', th, s);
  return P;
}

// ---------- ③ loss curve (MP4 only) ----------

// Loss history drawn up to step `upto` (the current frame), on axes fixed for the whole run
export function drawLoss(ctx, r, run, upto, th, s = 1, opts = {}) {
  const log = !!opts.log;
  const legendH = 26 * s;
  const P = { x: r.x + 54 * s, y: r.y + 6 * s, w: r.w - 64 * s, h: r.h - 30 * s - legendH };
  const ok = v => v != null && Number.isFinite(v) && (!log || v > 0);
  const tr = v => (log ? Math.log10(v) : v);

  const { steps, train, val } = run.loss;
  const xMax = Math.max(1, run.meta.total_steps);
  const vals = [...train, ...val].filter(ok).map(tr);
  let [lo, hi] = vals.length ? [Math.min(...vals), Math.max(...vals)] : [0, 1];
  if (!log) lo = Math.min(0, lo);
  if (hi - lo < 1e-9) { hi += 0.5; if (log) lo -= 0.5; }
  const m = (hi - lo) * 0.06;
  if (log) lo -= m;
  hi += m;
  const X = v => P.x + v / xMax * P.w;
  const Y = v => P.y + P.h - (tr(v) - lo) / (hi - lo) * P.h;
  const Yt = v => P.y + P.h - (v - lo) / (hi - lo) * P.h;

  panel(ctx, P, th, s);
  let yTicks = niceTicks(lo, hi, 4);
  if (log && hi - lo >= 2) yTicks = yTicks.filter(Number.isInteger);
  grid(ctx, P, th, s, niceTicks(0, xMax, 6), yTicks, X, Yt, fmt, v => (log ? fmt(10 ** v) : fmt(v)));
  ctx.save();
  ctx.translate(r.x + 12 * s, P.y + P.h / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = th.text; ctx.font = font(12 * s); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(t('Loss') + (log ? ' (log)' : ''), 0, 0);
  ctx.restore();

  let n = 0;
  while (n < steps.length && steps[n] <= upto) n++;
  ctx.save();
  ctx.beginPath(); ctx.rect(P.x, P.y, P.w, P.h); ctx.clip();
  ctx.lineJoin = 'round';
  for (const [series, color] of [[train, th.train], [val, th.val]]) {
    ctx.strokeStyle = color; ctx.lineWidth = 3 * s;
    polyline(ctx, steps.slice(0, n), series.slice(0, n).map(v => (ok(v) ? v : null)), X, Y);
    const last = series[n - 1];
    if (n && ok(last)) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(X(steps[n - 1]), Y(last), 5 * s, 0, 7); ctx.fill(); }
  }
  ctx.restore();

  ctx.fillStyle = th.muted; ctx.font = font(12.5 * s); ctx.textAlign = 'right'; ctx.textBaseline = 'top';
  ctx.fillText(`${t('Train')}: ${fmtLoss(n ? train[n - 1] : null)}`, P.x + P.w - 10 * s, P.y + 8 * s);
  ctx.fillText(`${t('Valid')}: ${fmtLoss(n ? val[n - 1] : null)}`, P.x + P.w - 10 * s, P.y + 26 * s);

  legendRow(ctx, [
    { color: th.train, label: t('Training loss') },
    { color: th.val, label: t('Validation loss') },
  ], P.x + P.w / 2, P.y + P.h + 24 * s + legendH / 2, th, s);
}

// Canvas sized to its CSS box at device pixel ratio; returns the context in CSS pixels
export function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const { width, height } = canvas.getBoundingClientRect();
  const W = Math.max(1, Math.round(width * dpr)), H = Math.max(1, Math.round(height * dpr));
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: width, h: height };
}
