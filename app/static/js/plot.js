// Canvas drawing shared by the screen and the MP4 export.
// Every function draws into a rectangle {x, y, w, h} of a 2D context; `s` scales fonts and lines.
import { t } from './i18n.js';
import { configBadge, stepLabel } from './describe.js';

export const FONT = 'system-ui, -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", "Segoe UI", sans-serif';
export const MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace';

const SERIES = {
  train: '#e4574e', val: '#f2a33a', pred: '#3b7ddd', truth: '#6b7280',
};
const SLICES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'];   // the four weights of the loss landscape
export const LIGHT = {
  ...SERIES, bg: '#ffffff', panel: '#fbfcfd', grid: '#e3e8ef', frame: '#cfd6df',
  text: '#111827', muted: '#5b6472', badge: 'rgba(255,255,255,.92)', slices: SLICES,
};

export function screenTheme() {
  const cs = getComputedStyle(document.documentElement);
  const v = (name, fallback) => cs.getPropertyValue(name).trim() || fallback;
  return {
    ...SERIES, bg: v('--surface', LIGHT.bg), panel: v('--plot-panel', LIGHT.panel), grid: v('--plot-grid', LIGHT.grid),
    frame: v('--plot-frame', LIGHT.frame), text: v('--text', LIGHT.text), muted: v('--muted', LIGHT.muted),
    badge: v('--plot-badge', LIGHT.badge), slices: SLICES.map((c, i) => v(`--slice-${i + 1}`, c)),
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

function drawPrediction2D(ctx, r, m, pred, th, s, view, withPred, small = false) {   // small: no axis names or legend
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

  if (small) return P;
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

// ---------- loss landscape of one-input linear regression (landscape.js computes L) ----------

// Surface = training loss of every (w, b); line = the path of the run, solid up to frame `idx`
export function drawLandscape(ctx, r, L, idx, run, th, s = 1, opts = {}) {
  const view = opts.view || DEFAULT_VIEW, log = !!opts.log;
  const tr = v => (log ? Math.log10(Math.max(v, 1e-12)) : v);
  const { ws, bs, z } = L, n1 = ws.length, n2 = bs.length;
  const zlo = tr(L.zMin), zhi = tr(L.zMax) + 1e-12;
  const P = { x: r.x + 4 * s, y: r.y + 4 * s, w: r.w - 8 * s, h: r.h - 8 * s };
  panel(ctx, P, th, s);

  const u = v => (v - ws[0]) / (ws[n1 - 1] - ws[0]) * 2 - 1;
  const v_ = v => (v - bs[0]) / (bs[n2 - 1] - bs[0]) * 2 - 1;
  const h = v => ((tr(v) - zlo) / (zhi - zlo) * 2 - 1) * 0.72;
  const ca = Math.cos(view.az), sa = Math.sin(view.az), ce = Math.cos(view.el), se = Math.sin(view.el);
  const scale = Math.min(P.w / 3.4, P.h / 3.0);
  const cx = P.x + P.w / 2, cy = P.y + P.h * 0.5;
  const project = (a, b, c) => {
    const xr = a * ca - b * sa, yr = a * sa + b * ca;
    return [cx + xr * scale, cy - (yr * se + c * ce) * scale, -yr * ce + c * se];
  };

  // floor and back edge of the box
  ctx.strokeStyle = th.frame; ctx.lineWidth = 1.2 * s;
  const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
  const floor = corners.map(([a, b]) => project(a, b, -0.72));
  ctx.beginPath(); floor.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); ctx.stroke();
  const back = floor.reduce((best, p, i) => (p[2] < floor[best][2] ? i : best), 0);
  const [tx, ty] = project(...corners[back], 0.72);
  ctx.beginPath(); ctx.moveTo(floor[back][0], floor[back][1]); ctx.lineTo(tx, ty); ctx.stroke();

  // the surface, far to near
  const quads = [];
  for (let j = 0; j < n2 - 1; j++) for (let i = 0; i < n1 - 1; i++) {
    const zs = [z[j * n1 + i], z[j * n1 + i + 1], z[(j + 1) * n1 + i + 1], z[(j + 1) * n1 + i]];
    const pts = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]].map(([a, b], k) => project(u(ws[a]), v_(bs[b]), h(zs[k])));
    const zm = (tr(zs[0]) + tr(zs[1]) + tr(zs[2]) + tr(zs[3])) / 4;
    quads.push({ d: (pts[0][2] + pts[1][2] + pts[2][2] + pts[3][2]) / 4, pts, color: coolwarm((zm - zlo) / (zhi - zlo)) });
  }
  quads.sort((a, b) => a.d - b.d);
  ctx.save();
  ctx.beginPath(); ctx.rect(P.x, P.y, P.w, P.h); ctx.clip();
  ctx.lineWidth = 0.6 * s;
  for (const q of quads) {
    ctx.fillStyle = ctx.strokeStyle = q.color;
    ctx.beginPath(); q.pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath();
    ctx.globalAlpha = 0.82; ctx.fill(); ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // the path on top: solid up to the current frame, faint after it
  const pts = L.path.map(p => ({ ...p, xy: project(u(p.w), v_(p.b), h(p.loss)) }));
  let cur = 0;
  while (cur + 1 < pts.length && pts[cur + 1].frame <= idx) cur++;
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  const line = (from, to, alpha, width) => {
    if (to <= from) return;
    ctx.globalAlpha = alpha; ctx.strokeStyle = th.pred; ctx.lineWidth = width * s;
    ctx.beginPath(); for (let k = from; k <= to; k++) (k === from ? ctx.moveTo : ctx.lineTo).call(ctx, pts[k].xy[0], pts[k].xy[1]);
    ctx.stroke();
  };
  line(cur, pts.length - 1, 0.25, 2);
  line(0, cur, 1, 2.6);
  ctx.globalAlpha = 1;
  const dot = (p, rad, fill, stroke) => {
    ctx.beginPath(); ctx.arc(p.xy[0], p.xy[1], rad * s, 0, 7);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2 * s; ctx.stroke(); }
  };
  dot(pts[0], 5, th.bg, th.pred);                         // start
  const now = pts[cur];
  const [fx, fy] = project(u(now.w), v_(now.b), -0.72);   // drop line to the floor, for depth
  ctx.strokeStyle = th.pred; ctx.lineWidth = 1.2 * s; ctx.setLineDash([4 * s, 4 * s]);
  ctx.beginPath(); ctx.moveTo(now.xy[0], now.xy[1]); ctx.lineTo(fx, fy); ctx.stroke(); ctx.setLineDash([]);
  ctx.beginPath(); ctx.arc(fx, fy, 2.5 * s, 0, 7); ctx.fillStyle = th.pred; ctx.fill();
  dot(now, 7, th.pred, th.bg);
  ctx.restore();

  if (small) return P;
  // axis names at the middle of the two front floor edges
  ctx.fillStyle = th.muted; ctx.font = font(13 * s); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  const lab = (a, b, text) => { const [x, y] = project(a, b, -0.72); ctx.fillText(text, x, y + 14 * s); };
  lab(0, ca > 0 ? -1.12 : 1.12, t('Slope w'));
  lab(sa > 0 ? 1.12 : -1.12, 0, t('Intercept b'));

  const total = run.meta.total_steps;
  badge(ctx, [
    `${stepLabel(run.meta.model)} ${now.step.toLocaleString()}/${total.toLocaleString()}`,
    `ŷ = ${fmt(now.w)}·x ${now.b < 0 ? '−' : '+'} ${fmt(Math.abs(now.b))}`,
    `${t('Training loss')} ${fmtLoss(now.loss)}${log ? ' (log)' : ''}`,
  ], P.x + 10 * s, P.y + 10 * s, 'left', th, s, FONT, 13);
  legend(ctx, [
    { color: coolwarm(0.8), width: 6, label: t('Loss of every line') },
    { color: th.pred, width: 2.6, label: t('Path of this run') },
  ], P.x + P.w - 10 * s, P.y + P.h - 10 * s, 'right', th, s);
}

// ---------- loss landscape of other linear and neural network runs (landscape.js computes S) ----------

// The few weights that lowered the loss the most: where they sit in the network, and for each the training
// loss along it with every other weight where training ended (a dashed slice through the end point, fading
// away from it) and the slope that weight came down (solid: faint for the whole run, full up to frame `idx`,
// where the point is; open circles: start and end). Runs saved without the paths show an arrow from the
// start to the end instead. Every slice has a handle at its end point, and the prediction is drawn small
// beside the network (two inputs: a surface seen as opts.view): at frame `idx`, or with opts.edit (tryWeight
// in landscape.js), a weight moved by hand, the one that makes.
// Returns where the panels and the reset button are, for the pointer: { panels: [{ x, y, w, h, x0, x1 }], reset }.
export function drawSlices(ctx, r, S, idx, run, th, s = 1, opts = {}) {
  const names = run.meta.feature_names, edit = opts.edit;
  const frame = run.frames[Math.max(0, Math.min(run.frames.length - 1, idx))];
  const pad = 10 * s, netH = Math.min(185 * s, Math.max(130 * s, r.h * 0.3)), netW = (r.w - 2 * pad) * 0.55;
  const layout = { panels: [], reset: null };
  drawWeightNet(ctx, { x: r.x + pad, y: r.y + 2 * s, w: netW, h: netH }, S, names, th, s);   // the network, and beside it the prediction
  const R = { x: r.x + pad + netW + 12 * s, y: r.y + 8 * s, w: r.w - 2 * pad - netW - 12 * s, h: netH - 12 * s };
  layout.reset = drawTriedPrediction(ctx, R, run, S, frame, edit, opts.view || DEFAULT_VIEW, th, s);

  // one line of numbers under the network
  const top4 = S.slices.reduce((a, sl) => a + (sl.share ?? 0), 0);
  const parts = [`${t('Training loss')} ${fmtLoss(S.start)} → ${fmtLoss(S.loss)}`];
  if (S.total > 0 && S.slices.length < S.n) parts.push(t('these {k}: {p} of the drop', { k: S.slices.length, p: pct(top4) }));
  if (S.idle) parts.push(t('{n} of {total} parameters barely changed the loss', { n: S.idle, total: S.n }));
  ctx.fillStyle = th.muted; ctx.font = font(12 * s); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(parts.join('  ·  '), r.x + r.w / 2, r.y + netH + 12 * s, r.w - 2 * pad);

  // the loss axis is the same in every panel, so the slopes compare; it fits the slopes came down, and
  // the slices (steeper: the other weights had adapted to the end) run off the top
  const came = S.slices.flatMap(sl => (sl.track || []).map(p => p.y)).filter(Number.isFinite);
  const cap = Number.isFinite(S.start) ? 1.5 * Math.max(S.start, S.loss) : Infinity;   // steep walls run off the top
  const top = came.length ? 1.4 * Math.max(...came) : Math.max(...S.slices.flatMap(sl => sl.ys).filter(Number.isFinite));
  const hi = Math.min(Math.max(top, S.loss), cap) * 1.06, yTicks = niceTicks(0, hi, 4);
  const epochNow = frame.step;
  const cols = S.slices.length > 1 ? 2 : 1, rows = Math.ceil(S.slices.length / cols);
  const left = 46 * s, gx = 16 * s, gy = 12 * s, titleH = 24 * s, tickH = 20 * s;
  const y0 = r.y + netH + 28 * s, cw = (r.w - left - pad - (cols - 1) * gx) / cols;
  const ch = (r.y + r.h - y0 - (rows - 1) * gy) / rows;
  S.slices.forEach((sl, k) => {
    const color = th.slices[k], cx = r.x + left + (k % cols) * (cw + gx), cy = y0 + Math.floor(k / cols) * (ch + gy);
    const P = { x: cx, y: cy + titleH, w: cw, h: ch - titleH - tickH };
    layout.panels.push({ ...P, x0: sl.xs[0], x1: sl.xs[sl.xs.length - 1] });

    numberBadge(ctx, cx + 9 * s, cy + 10 * s, k + 1, color, th, s);
    ctx.font = font(12 * s); ctx.fillStyle = th.muted; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    const share = sl.share == null ? '' : t('{p} of the drop', { p: pct(sl.share) });
    ctx.fillText(share, cx + cw, cy + 10 * s);
    const shareW = share ? ctx.measureText(share).width + 10 * s : 0;
    ctx.font = font(12.5 * s, 600); ctx.fillStyle = th.text; ctx.textAlign = 'left';
    ctx.fillText(paramName(S.sizes, sl.info, names), cx + 23 * s, cy + 10 * s, Math.max(10, cw - 23 * s - shareW));

    panel(ctx, P, th, s);
    const x0 = sl.xs[0], x1 = sl.xs[sl.xs.length - 1];
    const X = v => P.x + ((v - x0) / (x1 - x0)) * P.w;
    const Yt = v => P.y + P.h - (v / hi) * P.h;
    const Y = v => Math.max(P.y - P.h, Math.min(P.y + 2 * P.h, Yt(v)));
    grid(ctx, P, th, s, niceTicks(x0, x1, cw > 240 * s ? 4 : 3), k % cols ? [] : yTicks, X, Yt);
    if (k % cols) {                       // the right column shares the left one's loss labels
      ctx.strokeStyle = th.grid; ctx.lineWidth = 1 * s; ctx.beginPath();
      for (const v of yTicks) { ctx.moveTo(P.x, Yt(v)); ctx.lineTo(P.x + P.w, Yt(v)); }
      ctx.stroke();
    }

    ctx.save();
    ctx.beginPath(); ctx.rect(P.x, P.y, P.w, P.h); ctx.clip();
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    const fade = ctx.createLinearGradient(P.x, 0, P.x + P.w, 0);
    fade.addColorStop(0, withAlpha(color, 0.15)); fade.addColorStop(0.5, color); fade.addColorStop(1, withAlpha(color, 0.15));
    ctx.strokeStyle = fade; ctx.lineWidth = 2.4 * s;
    ctx.setLineDash([6 * s, 4 * s]);
    polyline(ctx, sl.xs, sl.ys, X, Y);
    ctx.setLineDash([]);

    const startLabel = (x, y) => {           // above the point, or below it at the top of the panel
      const below = y - 20 * s < P.y;
      ctx.fillStyle = th.muted; ctx.font = font(10.5 * s); ctx.textBaseline = below ? 'top' : 'bottom';
      ctx.textAlign = x < P.x + 24 * s ? 'left' : x > P.x + P.w - 24 * s ? 'right' : 'center';
      ctx.fillText(t('start'), x, below ? y + 7 * s : y - 6 * s);
    };
    const ring = (x, y) => {
      ctx.beginPath(); ctx.arc(x, y, 4 * s, 0, 7);
      ctx.fillStyle = th.panel; ctx.fill(); ctx.strokeStyle = color; ctx.lineWidth = 1.8 * s; ctx.stroke();
    };
    const dot = (x, y) => {
      ctx.beginPath(); ctx.arc(x, y, 6 * s, 0, 7);
      ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = th.bg; ctx.lineWidth = 2 * s; ctx.stroke();
    };
    const inside = y => Math.max(P.y, Math.min(P.y + P.h, y));    // a point off the top stays on the edge

    if (sl.track) {
      let now = 0;                       // the last point at or before the frame's epoch
      while (now + 1 < sl.track.length && sl.track[now + 1].epoch <= epochNow) now++;
      const line = (to, width, stroke) => {
        ctx.strokeStyle = stroke; ctx.lineWidth = width * s;
        ctx.beginPath();
        for (let k = 0; k <= to; k++) (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(sl.track[k].x), Y(sl.track[k].y));
        ctx.stroke();
      };
      line(sl.track.length - 1, 1.6, withAlpha(color, 0.35));
      line(now, 2.6, color);
      ctx.restore();

      const first = sl.track[0], last = sl.track[sl.track.length - 1], cur = sl.track[now];
      ring(X(first.x), inside(Y(first.y)));
      startLabel(X(first.x), inside(Y(first.y)));
      if (now < sl.track.length - 1) ring(X(last.x), inside(Y(last.y)));
      dot(X(cur.x), inside(Y(cur.y)));
    } else {
      // how training moved this weight: from its start value to where it ended
      const ya = P.y + P.h - 11 * s, xa = X(sl.init), xb = X(sl.final);
      ctx.strokeStyle = ctx.fillStyle = color; ctx.lineWidth = 1.8 * s;
      if (Math.abs(xb - xa) > 8 * s) {
        const dir = Math.sign(xb - xa);
        ctx.beginPath(); ctx.moveTo(xa + dir * 4 * s, ya); ctx.lineTo(xb - dir * 2 * s, ya); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(xb, ya); ctx.lineTo(xb - dir * 7 * s, ya - 4 * s); ctx.lineTo(xb - dir * 7 * s, ya + 4 * s); ctx.closePath(); ctx.fill();
      }
      ring(xa, ya);
      startLabel(xa, ya);
      ctx.restore();
      dot(xb, inside(Y(S.loss)));                                // the end point: where this slice was cut
    }

    // the handle: at the end point, or where it was moved to along the slice
    const moved = edit?.k === k, hx = X(moved ? edit.x : sl.final), hy = inside(Y(moved ? edit.loss : S.loss));
    if (moved) {
      ctx.strokeStyle = th.muted; ctx.lineWidth = 1 * s; ctx.setLineDash([3 * s, 3 * s]);
      ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(hx, P.y + P.h); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.beginPath(); ctx.arc(hx, hy, 8 * s, 0, 7);
    ctx.fillStyle = th.bg; ctx.fill(); ctx.strokeStyle = color; ctx.lineWidth = 2.5 * s; ctx.stroke();
    ctx.beginPath(); ctx.arc(hx, hy, 3 * s, 0, 7); ctx.fillStyle = color; ctx.fill();
  });
  return layout;
}

// The prediction small, on the same axes as the prediction plot: at this frame, or with a weight moved by
// hand (edit) the one that makes, in that weight's color over the one at the end (two inputs: just that
// surface). Returns the reset button's box while moved.
function drawTriedPrediction(ctx, R, run, S, frame, edit, view, th, s) {
  const m = run.meta, P = R;
  if (m.n_inputs === 2) drawPrediction2D(ctx, P, { ...m, y_true: null }, edit ? edit.pred : frame.pred, th, s, view, true, true);
  else drawTriedLine(ctx, P, m, S, frame, edit, th, s);
  return drawTriedBadges(ctx, P, m, frame, edit, th, s);
}

function drawTriedLine(ctx, P, m, S, frame, edit, th, s) {
  const xs = m.axes[0], [ylo, yhi] = yRange(m), xlo = xs[0], xhi = xs[xs.length - 1];
  const X = v => P.x + ((v - xlo) / (xhi - xlo || 1)) * P.w, Y = v => P.y + P.h - ((v - ylo) / (yhi - ylo)) * P.h;
  panel(ctx, P, th, s);
  ctx.save();
  ctx.beginPath(); ctx.rect(P.x, P.y, P.w, P.h); ctx.clip();
  ctx.globalAlpha = 0.45; ctx.fillStyle = th.train;
  m.x_train[0].forEach((x, i) => { ctx.beginPath(); ctx.arc(X(x), Y(m.y_train[i]), 1.8 * s, 0, 7); ctx.fill(); });
  ctx.globalAlpha = 1; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  if (edit) {
    ctx.strokeStyle = th.muted; ctx.lineWidth = 1.4 * s; polyline(ctx, xs, S.finalPred, X, Y);
    ctx.strokeStyle = th.slices[edit.k]; ctx.lineWidth = 2.8 * s; polyline(ctx, xs, edit.pred, X, Y);
  } else {
    ctx.strokeStyle = th.pred; ctx.lineWidth = 2.4 * s; polyline(ctx, xs, frame.pred, X, Y);
  }
  ctx.restore();
}

// which prediction it is, and the reset button while a weight is moved (its box is returned)
function drawTriedBadges(ctx, P, m, frame, edit, th, s) {
  const lines = edit ? [`${t('Changed by hand')} · ${t('Training loss')} ${fmtLoss(edit.loss)}`]
    : [`${stepLabel(m.model)} ${frame.step.toLocaleString()} · ${t('Training loss')} ${fmtLoss(frame.train)}`, t('Drag a handle on a dashed line')];
  badge(ctx, lines, P.x + 6 * s, P.y + 6 * s, 'left', th, s, FONT, 11);
  if (!edit) return null;
  const label = `↺ ${t('Reset')}`;
  ctx.font = font(11.5 * s, 600);
  const w = ctx.measureText(label).width + 16 * s, h = 22 * s, box = { x: P.x + P.w - w - 6 * s, y: P.y + P.h - h - 6 * s, w, h };
  ctx.fillStyle = th.badge; ctx.strokeStyle = th.frame; ctx.lineWidth = 1.2 * s;
  roundRect(ctx, box.x, box.y, w, h, 5 * s); ctx.fill(); ctx.stroke();
  ctx.fillStyle = th.text; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(label, box.x + w / 2, box.y + h / 2 + 0.5 * s);
  return box;
}

const pct = v => `${Math.round(v * 100)}%`;

function withAlpha(color, a) {
  const m = /^#([0-9a-f]{6})$/i.exec(color.trim());
  if (!m) return color;
  const n = parseInt(m[1], 16);
  return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

// a numbered circle: the slice's number in text color inside a ring of its color
function numberBadge(ctx, x, y, k, color, th, s) {
  ctx.beginPath(); ctx.arc(x, y, 8.5 * s, 0, 7);
  ctx.fillStyle = th.bg; ctx.fill();
  ctx.strokeStyle = color; ctx.lineWidth = 2.5 * s; ctx.stroke();
  ctx.fillStyle = th.text; ctx.font = font(10.5 * s, 700); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(String(k), x, y + 0.5 * s);
}

// neuron j of layer l (0 = the inputs): "x", "Hidden 1·3" or "ŷ"
function neuronName(sizes, l, j, names) {
  if (l === 0) return names?.[j] ?? `x${j + 1}`;
  if (l === sizes.length - 1) return 'ŷ';
  return `${t('Hidden {i}', { i: l })}·${j + 1}`;
}

export function paramName(sizes, info, names) {
  const to = neuronName(sizes, info.layer + 1, info.to, names);
  return info.bias ? t('Bias of {node}', { node: to })
    : t('Weight {from} → {to}', { from: neuronName(sizes, info.layer, info.from, names), to });
}

// The network with the sliced weights marked: a weight is its colored edge, a bias a ring round its neuron
function drawWeightNet(ctx, R, S, names, th, s) {
  const { sizes, slices } = S, MAX = 8;
  const shown = sizes.map((n, l) => {       // the neurons drawn: those the slices touch, then the first ones
    const need = new Set();
    for (const { info } of slices) {
      if (info.layer + 1 === l) need.add(info.to);
      if (!info.bias && info.layer === l) need.add(info.from);
    }
    for (let j = 0; need.size < Math.min(n, MAX); j++) need.add(j);
    return [...need].sort((a, b) => a - b);
  });
  ctx.font = font(12 * s);
  const leftW = Math.max(...shown[0].map(j => ctx.measureText(neuronName(sizes, 0, j, names)).width)) + 18 * s;
  const span = Math.min(R.w - leftW - 30 * s, 150 * s * (sizes.length - 1));
  const x0 = R.x + leftW + (R.w - leftW - 30 * s - span) / 2;
  const xs = sizes.map((_, l) => x0 + (span * l) / Math.max(1, sizes.length - 1));
  const top = R.y + 12 * s, bottom = R.y + R.h - 26 * s;
  const gap = Math.min(24 * s, (bottom - top) / Math.max(1, Math.max(...shown.map(v => v.length)) - 1));
  const rad = Math.max(4 * s, Math.min(9 * s, gap / 2 - 2 * s));
  const pos = shown.map((js, l) => {
    const y1 = (top + bottom) / 2 - (gap * (js.length - 1)) / 2;
    return new Map(js.map((j, i) => [j, [xs[l], y1 + i * gap]]));
  });

  ctx.strokeStyle = th.frame; ctx.lineWidth = 1 * s;
  ctx.beginPath();
  for (let l = 0; l < sizes.length - 1; l++) {
    for (const [, [ax, ay]] of pos[l]) for (const [, [bx, by]] of pos[l + 1]) { ctx.moveTo(ax, ay); ctx.lineTo(bx, by); }
  }
  ctx.stroke();
  const badges = [];
  slices.forEach(({ info }, k) => {
    const [bx, by] = pos[info.layer + 1].get(info.to);
    if (info.bias) {                      // above the output neuron, beside a hidden one (whose neighbours are close)
      badges.push(info.layer === sizes.length - 2 ? [bx, by - rad - 13 * s, k] : [bx + rad + 8 * s, by - rad - 5 * s, k]);
      return;
    }
    const [ax, ay] = pos[info.layer].get(info.from);
    ctx.strokeStyle = th.slices[k]; ctx.lineWidth = 3 * s;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    // the badge halfway along the edge, or further along where another badge is already
    const at = [0.5, 0.3, 0.7, 0.2, 0.8].map(u => [ax + u * (bx - ax), ay + u * (by - ay)]);
    const free = at.find(([x, y]) => badges.every(([x2, y2]) => Math.hypot(x - x2, y - y2) > 19 * s)) || at[0];
    badges.push([...free, k]);
  });

  sizes.forEach((n, l) => {
    for (const [j, [x, y]] of pos[l]) {
      ctx.beginPath(); ctx.arc(x, y, rad, 0, 7);
      ctx.fillStyle = th.panel; ctx.fill();
      ctx.strokeStyle = th.muted; ctx.lineWidth = 1.2 * s; ctx.stroke();
      ctx.fillStyle = th.muted; ctx.textBaseline = 'middle';
      if (l > 0 && l < sizes.length - 1 && rad >= 5 * s) {
        ctx.font = font(Math.min(9 * s, rad * 1.35)); ctx.textAlign = 'center'; ctx.fillText(String(j + 1), x, y + 0.5 * s);
      } else if (l === 0) {
        ctx.font = font(12 * s); ctx.fillStyle = th.text; ctx.textAlign = 'right'; ctx.fillText(neuronName(sizes, 0, j, names), x - rad - 6 * s, y);
      } else if (l === sizes.length - 1) {
        ctx.font = font(12 * s); ctx.fillStyle = th.text; ctx.textAlign = 'left'; ctx.fillText('ŷ', x + rad + 9 * s, y);
      }
    }
    if (n > shown[l].length) {
      ctx.fillStyle = th.muted; ctx.font = font(12 * s); ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      const last = [...pos[l].values()].at(-1);
      ctx.fillText(`⋮ ${n}`, xs[l] + rad + 4 * s, last[1] + gap * 0.6);
    }
    ctx.fillStyle = th.muted; ctx.font = font(11 * s); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    const name = l === 0 ? t('Input') : l === sizes.length - 1 ? t('Output') : t('Hidden {i}', { i: l });
    ctx.fillText(name, xs[l], R.y + R.h);
  });
  slices.forEach(({ info }, k) => {
    if (!info.bias) return;
    const [x, y] = pos[info.layer + 1].get(info.to);
    ctx.beginPath(); ctx.arc(x, y, rad + 3.5 * s, 0, 7);
    ctx.strokeStyle = th.slices[k]; ctx.lineWidth = 2.5 * s; ctx.stroke();
  });
  for (const [x, y, k] of badges) numberBadge(ctx, x, y, k + 1, th.slices[k], th, s);
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
