// node --test tests/js   (the loss landscape, app/static/js/landscape.js: the whole surface of one-input
// linear regression, and slices along single weights for other linear and neural network runs)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { train, trainingLoss } from '../../app/static/js/nn.js';
import { SLICES, hasLandscape, hasSlices, landscape, paramInfo, paramSlices } from '../../app/static/js/landscape.js';

const CFG = { n_inputs: 1, model: 'linear', hidden_layers: [], activation: 'relu', loss: 'mse', huber_delta: 1,
  optimizer: 'bgd', learning_rate: 0.1, epochs: 60, batch_method: 'bgd', batch_size: 32, seed: 3 };

function meta(f = x => 3 * x + 10) {      // like trainer.meta(): raw x and y, and the plot axis
  const xs = Array.from({ length: 100 }, (_, i) => -5 + (10 * i) / 99), noise = i => 0.5 * Math.sin(7 * i);
  const tr = xs.filter((_, i) => i % 5), va = xs.filter((_, i) => !(i % 5));
  return { n_inputs: 1, model: 'linear', x_train: [tr], y_train: tr.map((x, i) => f(x) + noise(i)), x_val: [va],
           y_val: va.map(f), axes: [Array.from({ length: 200 }, (_, i) => -5 + i / 19.9)], total_steps: 60 };
}

function run(cfg = {}, m = meta()) {
  const c = { ...CFG, ...cfg }, frames = [];
  let params = null;
  for (const ev of train(c, m)) {
    if (ev.type === 'frame') frames.push(ev);
    if (ev.type === 'params') params = ev;
  }
  return { config: c, meta: m, frames, params };
}

function meta2(f = (a, b) => Math.abs(a) + Math.sin(b)) {      // two inputs
  const n = 150, x1 = Array.from({ length: n }, (_, i) => -3 + (6 * i) / (n - 1)), x2 = x1.map((_, i) => 3 * Math.cos(i));
  const pick = keep => [x1.filter((_, i) => keep(i)), x2.filter((_, i) => keep(i))];
  const tr = pick(i => i % 5), va = pick(i => !(i % 5)), axis = Array.from({ length: 5 }, (_, i) => -3 + 1.5 * i);
  return { n_inputs: 2, feature_names: ['x1', 'x2'], x_train: tr, y_train: tr[0].map((a, i) => f(a, tr[1][i])),
           x_val: va, y_val: va[0].map((a, i) => f(a, va[1][i])), axes: [axis, axis], total_steps: 0 };
}

const NET = { model: 'neural_net', hidden_layers: [4, 4], activation: 'tanh', optimizer: 'adam', learning_rate: 0.02,
  batch_method: 'mini', batch_size: 16, epochs: 300 };

test('only one-input linear regression has a landscape', () => {
  const r = run();
  assert.ok(hasLandscape(r));
  assert.ok(!hasLandscape({ ...r, config: { ...r.config, model: 'neural_net' } }));
  assert.ok(!hasLandscape({ ...r, meta: { ...r.meta, n_inputs: 2 } }));
});

for (const loss of ['mse', 'mae', 'huber']) {
  test(`${loss}: the path read from the frames has the losses the run recorded`, () => {
    const r = run({ loss }), L = landscape(r);
    assert.equal(L.path.length, r.frames.length);
    for (const p of L.path) assert.ok(Math.abs(L.loss(p.w, p.b) - p.loss) <= 2e-3 * Math.max(1, p.loss), `${p.step}: ${L.loss(p.w, p.b)} vs ${p.loss}`);
  });
}

test('the run walks down to the bottom of the bowl (y = 3x + 10)', () => {
  const L = landscape(run({ epochs: 200 }));
  const end = L.path.at(-1), start = L.path[0];
  assert.ok(Math.abs(end.w - 3) < 0.1 && Math.abs(end.b - 10) < 0.2, `${end.w}, ${end.b}`);
  assert.ok(end.loss < start.loss);
  const i = L.z.indexOf(Math.min(...L.z)), wMin = L.ws[i % L.ws.length], bMin = L.bs[Math.floor(i / L.ws.length)];
  assert.ok(Math.abs(wMin - end.w) <= L.ws[1] - L.ws[0] && Math.abs(bMin - end.b) <= L.bs[1] - L.bs[0]);
});

// ---------- slices along single weights ----------

test('every other linear or neural network run with saved weights has slices', () => {
  const r = run(NET);
  assert.ok(hasSlices(r) && !hasLandscape(r));
  assert.ok(!hasSlices(run()));                                       // the whole surface instead
  assert.ok(hasSlices(run({ n_inputs: 2 }, meta2())));                 // linear with two inputs
  assert.ok(!hasSlices({ ...r, params: null }));                       // saved before weights were saved
  assert.ok(!hasSlices({ ...r, config: { ...r.config, model: 'decision_tree' } }));
});

test('the weights are named by layer and neuron, in the trainer order', () => {
  const sizes = [2, 3, 1];          // W0 3×2, b0 3, W1 1×3, b1 1
  assert.deepEqual(paramInfo(sizes, 0), { layer: 0, bias: false, to: 0, from: 0 });
  assert.deepEqual(paramInfo(sizes, 5), { layer: 0, bias: false, to: 2, from: 1 });
  assert.deepEqual(paramInfo(sizes, 6), { layer: 0, bias: true, to: 0, from: null });
  assert.deepEqual(paramInfo(sizes, 10), { layer: 1, bias: false, to: 0, from: 1 });
  assert.deepEqual(paramInfo(sizes, 12), { layer: 1, bias: true, to: 0, from: null });
  assert.equal(paramInfo(sizes, 13), null);
});

for (const [name, cfg] of [['adam', {}], ['plain gradient descent', { optimizer: 'bgd', batch_method: 'bgd', learning_rate: 0.3 }],
                           ['SGD', { optimizer: 'sgd', batch_method: 'sgd', learning_rate: 0.01, epochs: 40 }],
                           ['momentum, many epochs', { optimizer: 'momentum', learning_rate: 0.03, epochs: 1000 }]]) {
  test(`${name}: the saved weights give the losses of the run, and the contributions add up to its drop`, () => {
    const r = run({ ...NET, ...cfg }), { init, final, contrib } = r.params, lossAt = trainingLoss(r.config, r.meta);
    assert.equal(final.length, 4 + 4 + 16 + 4 + 4 + 1);
    const first = r.frames[0].train, last = r.frames.at(-1).train;
    assert.ok(Math.abs(lossAt(init) - first) < 1e-4 * first, `${lossAt(init)} vs ${first}`);
    assert.ok(Math.abs(lossAt(final) - last) < 1e-3 * last + 1e-6, `${lossAt(final)} vs ${last}`);
    const sum = contrib.reduce((a, c) => a + c, 0);
    assert.ok(Math.abs(sum - (first - last)) < 0.15 * (first - last), `sum ${sum} vs drop ${first - last}`);
  });
}

test('the slices: the largest contributions first, each through the end point and spanning the start', () => {
  const r = run(NET), S = paramSlices(r), { contrib } = r.params;
  assert.equal(S.slices.length, SLICES);
  const sorted = [...contrib].sort((a, b) => b - a);
  assert.deepEqual(S.slices.map(sl => sl.contrib), sorted.slice(0, SLICES));
  assert.ok(Math.abs(S.loss - r.frames.at(-1).train) < 1e-3 * S.loss);
  for (const sl of S.slices) {
    const mid = (sl.xs.length - 1) / 2;
    assert.ok(Math.abs(sl.xs[mid] - sl.final) < 1e-12 && Math.abs(sl.ys[mid] - S.loss) < 1e-12);
    assert.ok(sl.xs[0] < Math.min(sl.init, sl.final) && Math.max(sl.init, sl.final) < sl.xs.at(-1));
    assert.ok(Math.min(...sl.ys) > S.loss - 0.05, 'the end is near the bottom of each slice');
    assert.equal(sl.share, sl.contrib / S.total);
  }
  assert.equal(S.n, contrib.length);
});

test('the slope a weight came down: its drop is what it lowered the loss by, and it touches the slice at the end', () => {
  const r = run({ ...NET, optimizer: 'bgd', batch_method: 'bgd', learning_rate: 0.1, epochs: 600 }), S = paramSlices(r);
  const lossAt = trainingLoss(r.config, r.meta);
  for (const sl of S.slices) {
    assert.ok(Math.abs(sl.track[0].y - S.loss - sl.contrib) < 1e-5, `${sl.track[0].y - S.loss} vs ${sl.contrib}`);
    const [a, b] = sl.track.slice(-21, -20).concat(sl.track.slice(-1));        // over the last 20 frames
    const h = 1e-4, theta = Float64Array.from(r.params.final);
    theta[sl.index] += h; const up = lossAt(theta);
    theta[sl.index] -= 2 * h; const down = lossAt(theta);
    const came = (b.y - a.y) / (b.x - a.x), cut = (up - down) / (2 * h);
    assert.ok(Math.abs(came - cut) <= 0.3 * Math.abs(cut) + 1e-3, `slope ${came} vs slice ${cut}`);
  }
});

test('the paths of the top weights: every frame and more often early on, from the start to the end', () => {
  for (const cfg of [NET, { ...NET, optimizer: 'momentum', learning_rate: 0.05, epochs: 400 }]) {
    const r = run(cfg), S = paramSlices(r), { paths } = r.params;
    assert.deepEqual(paths.map(p => p.index), S.slices.map(sl => sl.index));
    const epochs = r.params.path_epochs;
    assert.deepEqual(epochs, [...epochs].sort((a, b) => a - b));
    for (const f of r.frames) assert.ok(epochs.includes(f.step), `frame ${f.step}`);
    assert.ok(epochs.length > r.frames.length && epochs[1] < 1, 'mini-batches: points within the first epoch too');
    for (const sl of S.slices) {
      assert.equal(sl.track.length, epochs.length);
      assert.ok(Math.abs(sl.track[0].x - sl.init) < 1e-5 && Math.abs(sl.track.at(-1).x - sl.final) < 1e-5);
      assert.ok(Math.abs(sl.track.at(-1).y - S.loss) < 1e-3 * S.loss, 'the last point is the end point');
      for (const p of sl.track) assert.ok(sl.xs[0] < p.x && p.x < sl.xs.at(-1), 'the slice spans the whole path');
    }
  }
});

test('momentum overshoots: the path goes past where it ends, and the slice shows it', () => {
  const r = run({ ...NET, optimizer: 'momentum', learning_rate: 0.05, epochs: 400 }), S = paramSlices(r);
  const over = S.slices.some(sl => {
    const xs = sl.track.map(p => p.x), lo = Math.min(sl.init, sl.final), hi = Math.max(sl.init, sl.final);
    return Math.min(...xs) < lo - 0.05 * (hi - lo) || Math.max(...xs) > hi + 0.05 * (hi - lo);
  });
  assert.ok(over);
});

test('runs saved without paths still have slices', () => {
  const r = run(NET), S = paramSlices({ ...r, params: { ...r.params, paths: undefined } });
  assert.ok(S.slices.every(sl => sl.track === null));
});

test('weights of dead ReLU neurons never move, change nothing, and are counted', () => {
  const r = run({ ...NET, activation: 'relu', seed: 42 }), S = paramSlices(r), { init, final, contrib } = r.params;
  const dead = contrib.map((c, i) => i).filter(i => contrib[i] === 0);
  assert.ok(dead.length > 0 && S.idle >= dead.length && S.idle < S.n, `${dead.length} dead, ${S.idle} idle of ${S.n}`);
  for (const i of dead) assert.equal(final[i], init[i]);
});

test('linear regression with two inputs: all three weights, by contribution', () => {
  const r = run({ n_inputs: 2, epochs: 100 }, meta2()), S = paramSlices(r);
  assert.equal(S.slices.length, 3);
  assert.deepEqual(S.sizes, [2, 1]);
  assert.deepEqual(new Set(S.slices.map(sl => sl.index)), new Set([0, 1, 2]));
});

test('a diverged run saves no weights', () => {
  const events = [...train({ ...CFG, ...NET, optimizer: 'sgd', batch_method: 'sgd', learning_rate: 50, epochs: 50 }, meta())];
  assert.equal(events.at(-1).status, 'diverged');
  assert.ok(!events.some(e => e.type === 'params'));
});
