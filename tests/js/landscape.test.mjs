// node --test tests/js   (the loss landscape of one-input linear regression, app/static/js/landscape.js)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { train } from '../../app/static/js/nn.js';
import { hasLandscape, landscape } from '../../app/static/js/landscape.js';

const CFG = { n_inputs: 1, model: 'linear', hidden_layers: [], activation: 'relu', loss: 'mse', huber_delta: 1,
  optimizer: 'bgd', learning_rate: 0.1, epochs: 60, batch_method: 'bgd', batch_size: 32, seed: 3 };

function meta(f = x => 3 * x + 10) {      // like trainer.meta(): raw x and y, and the plot axis
  const xs = Array.from({ length: 100 }, (_, i) => -5 + (10 * i) / 99), noise = i => 0.5 * Math.sin(7 * i);
  const tr = xs.filter((_, i) => i % 5), va = xs.filter((_, i) => !(i % 5));
  return { n_inputs: 1, model: 'linear', x_train: [tr], y_train: tr.map((x, i) => f(x) + noise(i)), x_val: [va],
           y_val: va.map(f), axes: [Array.from({ length: 200 }, (_, i) => -5 + i / 19.9)], total_steps: 60 };
}

function run(cfg = {}) {
  const c = { ...CFG, ...cfg }, m = meta(), frames = [];
  for (const ev of train(c, m)) if (ev.type === 'frame') frames.push(ev);
  return { config: c, meta: m, frames };
}

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
