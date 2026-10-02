// Linear regression with one input has only two weights, ŷ = w·x + b, so its whole loss landscape fits
// in a 3D plot: the training loss of every (w, b) as a surface, and the path this run took over it.
// The path is read back from the frames (each prediction is a straight line), so every saved linear run
// has it. w and b are in the data's own units; the loss is on standardized y, like everywhere in the app.
import { lossFns } from './nn.js';

const GRID = 41;            // surface resolution (GRID × GRID losses)

export const hasLandscape = run =>
  run?.config?.model === 'linear' && run.meta?.n_inputs === 1 && !!run.frames?.length;

function meanStd(v) {
  const n = v.length, m = v.reduce((s, x) => s + x, 0) / n;
  return [m, Math.max(Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / n), 1e-12)];
}

// { ws, bs, z (row = b index, column = w index), zMin, zMax, path: [{ w, b, loss, step, frame }] }
export function landscape(run) {
  const { meta, frames, config } = run;
  const xs = meta.x_train[0], ys = meta.y_train, n = ys.length;
  const [, sx] = meanStd(xs), [, sy] = meanStd(ys);
  const f = lossFns(config).f;
  const loss = (w, b) => {
    let s = 0;
    for (let i = 0; i < n; i++) s += f((w * xs[i] + b - ys[i]) / sy);
    return s / n;
  };

  // each frame's prediction is a line over the plot axis: its two ends give w and b
  const ax = meta.axes[0], x0 = ax[0], x1 = ax[ax.length - 1];
  const path = [];
  frames.forEach((fr, frame) => {
    const p0 = fr.pred[0], p1 = fr.pred[fr.pred.length - 1], w = (p1 - p0) / (x1 - x0), b = p0 - w * x0;
    if (fr.train != null && Number.isFinite(w) && Number.isFinite(b)) path.push({ w, b, loss: fr.train, step: fr.step, frame });
  });
  if (!path.length) return null;

  // the path, plus at least one standard unit around where it ended, so the bowl shows
  const end = path[path.length - 1];
  const span = (vals, c, unit) => {
    let lo = Math.min(...vals, c - unit), hi = Math.max(...vals, c + unit);
    const pad = (hi - lo) * 0.08;
    return [lo - pad, hi + pad];
  };
  const [wLo, wHi] = span(path.map(p => p.w), end.w, sy / sx);
  const [bLo, bHi] = span(path.map(p => p.b), end.b, sy);
  const axis = (lo, hi) => Array.from({ length: GRID }, (_, i) => lo + (hi - lo) * i / (GRID - 1));
  const ws = axis(wLo, wHi), bs = axis(bLo, bHi);
  const z = new Float64Array(GRID * GRID);
  let zMin = Infinity, zMax = -Infinity;
  for (let j = 0; j < GRID; j++) for (let i = 0; i < GRID; i++) {
    const v = loss(ws[i], bs[j]);
    z[j * GRID + i] = v;
    if (v < zMin) zMin = v;
    if (v > zMax) zMax = v;
  }
  for (const p of path) { zMin = Math.min(zMin, p.loss); zMax = Math.max(zMax, p.loss); }
  return { ws, bs, z, zMin, zMax, path, loss };
}
