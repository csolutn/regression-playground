// Linear regression and neural network training in the browser: the same ② → ③ → ④ loop as
// train_gradient() in app/ml/trainer.py, written out by hand so it needs no library.
//
// train(cfg, meta, opts) is a generator of the trainer's events ('start', 'frame', 'loss', 'params', 'end'),
// so the output panel draws a browser run exactly like a server run. The data comes from the server
// (POST /api/prepare), which builds it as before; only the random weights and batch order differ,
// so the numbers differ from the server's while the behaviour is the same.
// Runs in a Web Worker (train-worker.js); plain module, so `node --test` can run it too.

export const TRAINER_VERSION = 1;  // raise it when train() makes other numbers: saved runs are replayed from their settings
export const MAX_FRAMES = 300;      // one frame per epoch up to this many, then every k-th
export const LOSS_EVENTS = 100;     // about this many loss events per run
export const CHECKPOINT_GROWTH = 1.2;   // weight contributions: checkpoints after steps 1, 2, 3, 4, 5, 6, 8, 10, …, and every epoch
const MAX_SPLITS = 4;                   // … with a stretch between two of them halved at most this many times
export const TOP_WEIGHTS = 4;           // the weights with the largest contributions: their paths (values, contributions so far) are saved

// ---------- small helpers ----------

function mulberry32(seed) {          // seeded random numbers in [0, 1)
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const sig = (v, digits) => Number(v.toPrecision(digits));
const finite = v => (Number.isFinite(v) ? sig(v, 6) : null);

function erf(x) {                    // Numerical Recipes erfc, error < 1.2e-7
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? 1 - r : r - 1;
}

// f(z) and f'(z) given z and a = f(z), with PyTorch's defaults (LeakyReLU 0.01, ELU α = 1, exact GELU)
const ACTIVATIONS = {
  relu:       { f: z => (z > 0 ? z : 0),                  df: z => (z > 0 ? 1 : 0) },
  leaky_relu: { f: z => (z > 0 ? z : 0.01 * z),           df: z => (z > 0 ? 1 : 0.01) },
  elu:        { f: z => (z > 0 ? z : Math.expm1(z)),      df: (z, a) => (z > 0 ? 1 : a + 1) },
  gelu:       { f: z => 0.5 * z * (1 + erf(z / Math.SQRT2)),
                df: z => 0.5 * (1 + erf(z / Math.SQRT2)) + z * Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI) },
  tanh:       { f: Math.tanh,                             df: (z, a) => 1 - a * a },
  sigmoid:    { f: z => 1 / (1 + Math.exp(-z)),           df: (z, a) => a * (1 - a) },
  linear:     { f: z => z,                                df: () => 1 },
};

// loss value of one error e = pred − true, and its derivative (both averaged over the batch by the caller)
export function lossFns(cfg) {
  const d = cfg.huber_delta;
  return {
    mse:   { f: e => e * e,        df: e => 2 * e },
    mae:   { f: e => Math.abs(e),  df: e => Math.sign(e) },
    huber: { f: e => (Math.abs(e) < d ? 0.5 * e * e : d * (Math.abs(e) - 0.5 * d)),
             df: e => (Math.abs(e) < d ? e : d * Math.sign(e)) },
  }[cfg.loss];
}

export function structure(cfg) {
  return [cfg.n_inputs, ...(cfg.model === 'neural_net' ? cfg.hidden_layers : []), 1];
}

// ---------- the network ----------

class Network {
  constructor(sizes, activation, rand) {
    this.sizes = sizes;
    this.act = ACTIVATIONS[activation];
    this.params = [];                  // [W0, b0, W1, b1, …]; W is out × in, row-major
    for (let l = 0; l < sizes.length - 1; l++) {
      const nIn = sizes[l], nOut = sizes[l + 1], bound = 1 / Math.sqrt(nIn);   // nn.Linear's default init
      const uniform = n => Float64Array.from({ length: n }, () => (2 * rand() - 1) * bound);
      this.params.push(uniform(nOut * nIn), uniform(nOut));
    }
    this.grads = this.params.map(p => new Float64Array(p.length));
    this.capacity = 0;
  }

  // buffers for n samples: z (before the activation) and a (after it) for every layer, and deltas
  reserve(n) {
    if (n <= this.capacity) return;
    this.capacity = n;
    this.z = this.sizes.map(s => new Float64Array(n * s));
    this.a = this.sizes.map(s => new Float64Array(n * s));
    this.delta = this.sizes.map(s => new Float64Array(n * s));
  }

  // ② prediction for n samples in X (n × n_inputs, row-major); returns the output buffer
  forward(X, n) {
    this.reserve(n);
    const { sizes, params, act } = this, last = sizes.length - 2;
    this.a[0].set(X.subarray(0, n * sizes[0]));
    for (let l = 0; l <= last; l++) {
      const nIn = sizes[l], nOut = sizes[l + 1], W = params[2 * l], b = params[2 * l + 1];
      const A = this.a[l], Z = this.z[l + 1], out = this.a[l + 1];
      for (let i = 0; i < n; i++) {
        const ai = i * nIn, zi = i * nOut;
        let j = 0;
        for (; j + 3 < nOut; j += 4) {        // four outputs at a time: one read of x serves four sums
          let s0 = b[j], s1 = b[j + 1], s2 = b[j + 2], s3 = b[j + 3];
          const w0 = j * nIn, w1 = w0 + nIn, w2 = w1 + nIn, w3 = w2 + nIn;
          for (let k = 0; k < nIn; k++) {
            const x = A[ai + k];
            s0 += W[w0 + k] * x; s1 += W[w1 + k] * x; s2 += W[w2 + k] * x; s3 += W[w3 + k] * x;
          }
          Z[zi + j] = s0; Z[zi + j + 1] = s1; Z[zi + j + 2] = s2; Z[zi + j + 3] = s3;
        }
        for (; j < nOut; j++) {
          let s = b[j];
          const wj = j * nIn;
          for (let k = 0; k < nIn; k++) s += W[wj + k] * A[ai + k];
          Z[zi + j] = s;
        }
      }
      const m = n * nOut;
      if (l < last) for (let q = 0; q < m; q++) out[q] = act.f(Z[q]);   // no activation on the output layer
      else out.set(Z.subarray(0, m));
    }
    return this.a[last + 1];
  }

  // backpropagation after forward(): dOut[i] = dLoss/dpred for sample i; fills this.grads
  backward(dOut, n) {
    const { sizes, params, grads, act } = this;
    for (const g of grads) g.fill(0);
    let D = dOut;
    for (let l = sizes.length - 2; l >= 0; l--) {
      const nIn = sizes[l], nOut = sizes[l + 1], W = params[2 * l];
      const gW = grads[2 * l], gb = grads[2 * l + 1], A = this.a[l];
      let i = 0;
      for (; i + 3 < n; i += 4) {             // four samples at a time: one pass over gW instead of four
        const a0 = i * nIn, a1 = a0 + nIn, a2 = a1 + nIn, a3 = a2 + nIn;
        for (let j = 0; j < nOut; j++) {
          const d0 = D[i * nOut + j], d1 = D[(i + 1) * nOut + j], d2 = D[(i + 2) * nOut + j], d3 = D[(i + 3) * nOut + j];
          gb[j] += d0 + d1 + d2 + d3;
          const wj = j * nIn;
          for (let k = 0; k < nIn; k++) gW[wj + k] += d0 * A[a0 + k] + d1 * A[a1 + k] + d2 * A[a2 + k] + d3 * A[a3 + k];
        }
      }
      for (; i < n; i++) {
        const ai = i * nIn;
        for (let j = 0; j < nOut; j++) {
          const d = D[i * nOut + j];
          if (d === 0) continue;
          gb[j] += d;
          const wj = j * nIn;
          for (let k = 0; k < nIn; k++) gW[wj + k] += d * A[ai + k];
        }
      }
      if (l === 0) break;
      const P = this.delta[l], Z = this.z[l];
      P.fill(0, 0, n * nIn);
      for (let i = 0; i < n; i++) {
        const pi = i * nIn;
        for (let j = 0; j < nOut; j++) {
          const d = D[i * nOut + j];
          if (d === 0) continue;
          const wj = j * nIn;
          for (let k = 0; k < nIn; k++) P[pi + k] += d * W[wj + k];
        }
        for (let k = 0; k < nIn; k++) P[pi + k] *= act.df(Z[pi + k], A[pi + k]);
      }
      D = P;
    }
  }
}

// ④ optimization: the update rules of torch.optim with its default settings
function makeOptimizer(cfg, params, grads) {
  const lr = cfg.learning_rate, zeros = () => params.map(p => new Float64Array(p.length));
  const each = update => () => {
    for (let p = 0; p < params.length; p++) {
      const P = params[p], G = grads[p];
      for (let i = 0; i < P.length; i++) P[i] -= update(p, i, G[i]);
    }
  };
  if (cfg.optimizer === 'momentum') {
    const v = zeros();
    return each((p, i, g) => lr * (v[p][i] = 0.9 * v[p][i] + g));
  }
  if (cfg.optimizer === 'rmsprop') {
    const sq = zeros();
    return each((p, i, g) => {
      sq[p][i] = 0.99 * sq[p][i] + 0.01 * g * g;
      return lr * g / (Math.sqrt(sq[p][i]) + 1e-8);
    });
  }
  if (cfg.optimizer === 'adam') {
    const m = zeros(), v = zeros();
    let t = 0, c1 = 0, c2 = 0;
    const update = each((p, i, g) => {
      m[p][i] = 0.9 * m[p][i] + 0.1 * g;
      v[p][i] = 0.999 * v[p][i] + 0.001 * g * g;
      return (lr / c1) * m[p][i] / (Math.sqrt(v[p][i]) / Math.sqrt(c2) + 1e-8);
    });
    return () => { t++; c1 = 1 - 0.9 ** t; c2 = 1 - 0.999 ** t; update(); };
  }
  return each((p, i, g) => lr * g);    // BGD, SGD, mini-batch: plain gradient descent
}

// ---------- data from the server's meta ----------

function rows(columns) {             // [[x1…], [x2…]] → row-major Float64Array
  const n = columns[0].length, d = columns.length, X = new Float64Array(n * d);
  for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) X[i * d + k] = columns[k][i];
  return X;
}

function meanStd(values) {           // like numpy: population standard deviation
  const n = values.length, mean = values.reduce((s, v) => s + v, 0) / n;
  const std = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / n);
  return [mean, Math.max(std, 1e-12)];
}

// Standardized data, with statistics of the training data (as Dataset in app/ml/data.py)
export function prepareData(meta) {
  const d = meta.n_inputs, xs = meta.x_train.map(meanStd), [yMean, yStd] = meanStd(meta.y_train);
  const scaleX = X => { for (let i = 0; i < X.length; i++) X[i] = (X[i] - xs[i % d][0]) / xs[i % d][1]; return X; };
  const scaleY = y => Float64Array.from(y, v => (v - yMean) / yStd);
  let plot;
  if (d === 1) plot = [meta.axes[0]];
  else {                             // meshgrid: row = x2 index, column = x1 index
    const [a1, a2] = meta.axes;
    plot = [a2.flatMap(() => a1), a2.flatMap(v => a1.map(() => v))];
  }
  return {
    Xt: scaleX(rows(meta.x_train)), yt: scaleY(meta.y_train),
    Xv: scaleX(rows(meta.x_val)), yv: scaleY(meta.y_val),
    Xp: scaleX(rows(plot)), nPlot: plot[0].length,
    unscaleY: v => v * yStd + yMean, yMean,
  };
}

// A network that takes its weights flat, in train()'s order [W0, b0, W1, b1, …]
function flatNetwork(cfg) {
  const net = new Network(structure(cfg), cfg.activation, () => 0.5);
  net.load = flat => {
    let o = 0;
    for (const p of net.params) { for (let i = 0; i < p.length; i++) p[i] = flat[o + i]; o += p.length; }
    return net;
  };
  return net;
}

// The training loss (on standardized y, as train() reports it) of the network with the given flat
// weights: lossAt(flat) → number
export function trainingLoss(cfg, meta) {
  const { Xt, yt } = prepareData(meta), n = yt.length, loss = lossFns(cfg), net = flatNetwork(cfg);
  return flat => {
    const pred = net.load(flat).forward(Xt, n);
    let s = 0;
    for (let i = 0; i < n; i++) s += loss.f(pred[i] - yt[i]);
    return s / n;
  };
}

// Its prediction on the plot inputs, in the data's units (as a frame's pred): predictAt(flat) → [ŷ…]
export function prediction(cfg, meta) {
  const data = prepareData(meta), net = flatNetwork(cfg);
  return flat => Array.from(net.load(flat).forward(data.Xp, data.nPlot).subarray(0, data.nPlot), data.unscaleY);
}

// ---------- the training loop ----------

export function* train(cfg, meta, { timeLimitS = 30, now = () => performance.now() } = {}) {
  const started = now(), deadline = started + timeLimitS * 1000;
  const rand = mulberry32(cfg.seed), shuffleRand = mulberry32(cfg.seed ^ 0x5bd1e995);
  const data = prepareData(meta), { Xt, yt, Xv, yv, Xp } = data, d = meta.n_inputs;
  const nT = yt.length, nV = yv.length;
  const net = new Network(structure(cfg), cfg.activation, rand);
  const step = makeOptimizer(cfg, net.params, net.grads);
  const loss = lossFns(cfg);
  const bs = Math.min({ sgd: 1, mini: cfg.batch_size, bgd: nT }[cfg.batch_method], nT);
  const epochs = cfg.epochs;
  const frameEvery = Math.max(1, Math.ceil(epochs / MAX_FRAMES));
  const lossEvery = Math.max(1, Math.ceil(epochs / LOSS_EVENTS));
  const Xb = new Float64Array(bs * d), yb = new Float64Array(bs), dOut = new Float64Array(bs);
  const order = Uint32Array.from({ length: nT }, (_, i) => i);
  const history = { steps: [], train: [], val: [] };
  let pending = { steps: [], train: [], val: [] };

  const meanLoss = (X, y, n) => {
    const pred = net.forward(X, n);
    let s = 0;
    for (let i = 0; i < n; i++) s += loss.f(pred[i] - y[i]);
    return s / n;
  };

  // How much each weight lowered the training loss (Loss Change Allocation): between two checkpoints,
  // weight i moved by Δθᵢ while the full-batch gradient went from g to g', so it lowered the loss by
  // about −½(gᵢ + g'ᵢ)·Δθᵢ. Summed over the weights this should be the drop of the training loss between
  // them; where it is not (big steps across a curved valley), the stretch is halved and each half added
  // up the same way, down to MAX_SPLITS times.
  const nParams = net.params.reduce((s, p) => s + p.length, 0), dFull = new Float64Array(nT);
  const point = () => ({ p: new Float64Array(nParams), g: new Float64Array(nParams), l: 0 });
  const contrib = new Float64Array(nParams), halves = Array.from({ length: MAX_SPLITS }, point);
  let prev = point(), cur = point(), initParams = null, initLoss = 0;
  const setParams = flat => { let o = 0; for (const P of net.params) { P.set(flat.subarray(o, o + P.length)); o += P.length; } };
  // the full-batch gradient right after meanLoss(Xt, yt, nT) returned trainLoss (backward() reuses that forward pass)
  const gradient = (into, trainLoss) => {
    const pred = net.a[net.sizes.length - 1];
    for (let i = 0; i < nT; i++) dFull[i] = loss.df(pred[i] - yt[i]) / nT;
    net.backward(dFull, nT);
    let o = 0;
    for (let l = 0; l < net.params.length; l++) { into.p.set(net.params[l], o); into.g.set(net.grads[l], o); o += net.params[l].length; }
    into.l = trainLoss;
  };
  // the contributions along the straight line from point a to point b, if they add up to its drop within
  // tol (else the halves, each within tol / 2); true if it was split
  const along = (a, b, tol, depth = 0) => {
    let sum = 0;
    for (let i = 0; i < nParams; i++) sum -= 0.5 * (a.g[i] + b.g[i]) * (b.p[i] - a.p[i]);
    if (depth === MAX_SPLITS || Math.abs(sum - (a.l - b.l)) <= tol) {
      for (let i = 0; i < nParams; i++) contrib[i] -= 0.5 * (a.g[i] + b.g[i]) * (b.p[i] - a.p[i]);
      return false;
    }
    const m = halves[depth];
    for (let i = 0; i < nParams; i++) m.p[i] = 0.5 * (a.p[i] + b.p[i]);
    setParams(m.p);
    gradient(m, meanLoss(Xt, yt, nT));
    along(a, m, tol / 2, depth + 1);
    along(m, b, tol / 2, depth + 1);
    return true;
  };
  const checkpoint = tl => {            // tl: the training loss meanLoss(Xt, yt, nT) just returned
    gradient(cur, tl);
    if (!initParams) { initParams = cur.p.slice(); initLoss = tl; }
    else if (along(prev, cur, 0.02 * Math.abs(prev.l - tl) + 1e-5 * initLoss)) setParams(cur.p);   // back from halfway
    [prev, cur] = [cur, prev];
  };
  // every weight and its contribution so far, for the paths of the TOP_WEIGHTS: at every frame and,
  // where the weights move fast early on, at checkpoints after steps 1, 2, 3, 4, 5, 6, 8, 10, … too
  let steps = 0, nextCheckpoint = 1, nextRecord = 1;
  const snapshots = [];
  const record = epochAt => {           // right after a checkpoint; epochAt may be part of an epoch
    snapshots.push({ epoch: epochAt, w: Float32Array.from(net.params.flatMap(p => Array.from(p))), c: Float32Array.from(contrib) });
    nextRecord = Math.max(steps + 1, Math.ceil(steps * CHECKPOINT_GROWTH));
  };
  const frame = (epoch, tl, vl) => {
    const pred = net.forward(Xp, data.nPlot), out = new Array(data.nPlot);
    for (let i = 0; i < data.nPlot; i++) {
      const v = data.unscaleY(pred[i]);
      out[i] = sig(Number.isFinite(v) ? v : data.yMean, 4);
    }
    return { type: 'frame', step: epoch, pred: out, train: finite(tl), val: finite(vl) };
  };

  yield { type: 'start', meta };
  const vl0 = meanLoss(Xv, yv, nV), tl0 = meanLoss(Xt, yt, nT);
  checkpoint(tl0);
  record(0);
  yield frame(0, tl0, vl0);                                        // epoch 0: the untrained model

  let status = 'done', epoch = 0;
  while (epoch < epochs) {
    epoch++;
    for (let i = nT - 1; i > 0; i--) {                               // a new batch order every epoch
      const j = Math.floor(shuffleRand() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (let start = 0; start < nT; start += bs) {
      const n = Math.min(bs, nT - start);
      for (let i = 0; i < n; i++) {
        const r = order[start + i];
        for (let k = 0; k < d; k++) Xb[i * d + k] = Xt[r * d + k];
        yb[i] = yt[r];
      }
      const pred = net.forward(Xb, n);                               // ② prediction
      for (let i = 0; i < n; i++) dOut[i] = loss.df(pred[i] - yb[i]) / n;   // ③ loss (its gradient)
      net.backward(dOut, n);                                         //    backpropagation
      step();                                                        // ④ update the weights
      if (++steps >= nextCheckpoint) {     // the weights move fast early on: checkpoints in the epoch too
        nextCheckpoint = Math.ceil(steps * CHECKPOINT_GROWTH);
        if (start + bs < nT) {                                       // (the epoch's end has its own)
          checkpoint(meanLoss(Xt, yt, nT));
          if (steps >= nextRecord) record(epoch - 1 + (start + n) / nT);
        }
      }
    }

    const vl = meanLoss(Xv, yv, nV), tl = meanLoss(Xt, yt, nT);
    const diverged = !(Number.isFinite(tl) && Number.isFinite(vl));
    pending.steps.push(epoch);
    pending.train.push(finite(tl));
    pending.val.push(finite(vl));
    if (!diverged) checkpoint(tl);
    const last = epoch === epochs || diverged || now() > deadline, isFrame = epoch % frameEvery === 0 || last;
    if (!diverged && (isFrame || steps >= nextRecord)) record(epoch);
    if (isFrame) yield frame(epoch, tl, vl);
    if (epoch % lossEvery === 0 || last) {
      for (const k in history) history[k].push(...pending[k]);
      yield { type: 'loss', ...pending };
      pending = { steps: [], train: [], val: [] };
    }
    if (last) {
      if (diverged) status = 'diverged';
      else if (epoch < epochs) status = 'timeout';
      break;
    }
  }
  // the weights at the start and at the end (the last frame), for the loss landscape; not for a diverged run
  const out = a => Array.from(a, finite);
  if (status !== 'diverged' && prev.p.every(Number.isFinite) && contrib.every(Number.isFinite)) {
    const c = out(contrib), top = c.map((_, i) => i).sort((a, b) => c[b] - c[a]).slice(0, TOP_WEIGHTS);
    const paths = top.map(i => ({ index: i, values: snapshots.map(f => sig(f.w[i], 6)),
                                  lowered: snapshots.map(f => sig(f.c[i], 6)) }));
    yield { type: 'params', init: out(initParams), final: out(prev.p), contrib: c,
            path_epochs: snapshots.map(f => sig(f.epoch, 6)), paths };
  }
  yield { type: 'end', status, steps: epoch, final_train: history.train.at(-1), final_val: history.val.at(-1),
          duration: Math.round((now() - started) / 10) / 100 };
}
