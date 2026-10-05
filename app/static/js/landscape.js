// Linear regression with one input has only two weights, ŷ = w·x + b, so its whole loss landscape fits
// in a 3D plot: the training loss of every (w, b) as a surface, and the path this run took over it.
// The path is read back from the frames (each prediction is a straight line), so every saved linear run
// has it. w and b are in the data's own units; the loss is on standardized y, like everywhere in the app.
//
// Other linear and neural network runs have too many weights for that. For them the landscape is cut
// along one weight at a time (slices): the training loss as that weight changes while every other weight
// stays where training ended. It is drawn for the few weights that lowered the loss the most (the
// trainer adds up each weight's share, 'contrib'), from the weights the run saved (run.params).
//
// The slice is the landscape as it is at the end; while training it kept changing with the other weights.
// What gradient descent saw of it is only the slope under its feet, and those tangents joined up give
// the slope a weight came down: at every frame and, early on, more often (path_epochs) its value, and as
// height the end loss plus what it still lowered the loss by after then. It meets the slice at the end
// point, touching it there.
import { TOP_WEIGHTS, lossFns, prediction, structure, trainingLoss } from './nn.js';

const GRID = 41;            // surface resolution (GRID × GRID losses)
export const SLICES = TOP_WEIGHTS;   // slices drawn: the weights with the largest contributions
const SLICE_POINTS = 65;    // losses along each slice

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

// ---------- slices along single weights ----------

export const hasSlices = run => !hasLandscape(run) && ['linear', 'neural_net'].includes(run?.config?.model)
  && !!run.params?.final?.length && !!run.frames?.length;

// What weight i of the flat list [W0, b0, W1, b1, …] is (W is out × in): the layer it leads into
// (0 = the first after the inputs), whether it is a bias, and the neurons it joins (0-based)
export function paramInfo(sizes, i) {
  for (let l = 0; l < sizes.length - 1; l++) {
    const nIn = sizes[l], nOut = sizes[l + 1];
    if (i < nIn * nOut) return { layer: l, bias: false, to: Math.floor(i / nIn), from: i % nIn };
    i -= nIn * nOut;
    if (i < nOut) return { layer: l, bias: true, to: i, from: null };
    i -= nOut;
  }
  return null;
}

// { sizes, loss (at the end), start (loss before training), total (sum of contributions), idle (weights
//   that barely changed the loss), n, slices: [{ index, info, init, final, contrib, share, xs, ys,
//   track: [{ x, y, epoch }], the slope it came down (null without a path) }],
//   and to change a weight by hand (tryWeight): final, lossAt, predictAt, finalPred }
// Each slice spans the end value ± 1.25 × how far training moved it (at least ± 0.5), and the whole path.
export function paramSlices(run) {
  const { init, final, contrib, paths = [], path_epochs: epochs } = run.params;
  const pathOf = new Map(paths.map(p => [p.index, p]));
  const sizes = run.meta.structure || structure(run.config);
  const lossAt = trainingLoss(run.config, run.meta);
  const total = contrib.reduce((a, c) => a + c, 0), absTotal = contrib.reduce((a, c) => a + Math.abs(c), 0);
  const theta = Float64Array.from(final), loss = lossAt(theta);
  const top = contrib.map((_, i) => i).sort((a, b) => contrib[b] - contrib[a]).slice(0, SLICES);
  const slices = top.map(i => {
    const path = pathOf.get(i);
    const far = path ? Math.max(...path.values.map(v => Math.abs(v - final[i]))) * 1.1 : 0;
    const half = Math.max(1.25 * Math.abs(final[i] - init[i]), far, 0.5);
    const xs = Array.from({ length: SLICE_POINTS }, (_, k) => final[i] - half + (2 * half * k) / (SLICE_POINTS - 1));
    const ys = xs.map(x => { theta[i] = x; return lossAt(theta); });
    theta[i] = final[i];
    return { index: i, info: paramInfo(sizes, i), init: init[i], final: final[i], contrib: contrib[i],
             share: total > 0 ? contrib[i] / total : null, xs, ys,
             track: path ? path.values.map((x, k) => ({ x, y: loss + path.lowered.at(-1) - path.lowered[k], epoch: epochs[k] }))
               : null };
  });
  const predictAt = prediction(run.config, run.meta);
  return { sizes, slices, loss, start: run.frames[0].train, total, n: contrib.length,
           idle: contrib.filter(c => Math.abs(c) <= 1e-6 * absTotal).length,
           final: Float64Array.from(final), lossAt, predictAt, finalPred: predictAt(final) };
}

// Slice k's weight set to x by hand (kept within the slice), every other weight where training ended:
// { k, x, loss, pred } – a point on that slice, and the prediction it makes. Not a moment of the run.
export function tryWeight(S, k, x) {
  const sl = S.slices[k], theta = S.final.slice();
  x = Math.max(sl.xs[0], Math.min(sl.xs[sl.xs.length - 1], x));
  theta[sl.index] = x;
  return { k, x, loss: S.lossAt(theta), pred: S.predictAt(theta) };
}
