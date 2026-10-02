// node --test tests/js   (the browser trainer, app/static/js/nn.js)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { train } from '../../app/static/js/nn.js';

const BASE = {
  n_inputs: 1, model: 'neural_net', hidden_layers: [8], activation: 'tanh', loss: 'mse', huber_delta: 1,
  optimizer: 'adam', learning_rate: 0.02, epochs: 200, batch_method: 'mini', batch_size: 16, seed: 42,
};

function meta1d(f = Math.abs) {       // the shape of trainer.meta(): columns of x, and the plot axis
  const xs = Array.from({ length: 120 }, (_, i) => -5 + (10 * i) / 119);
  const train = xs.filter((_, i) => i % 5), val = xs.filter((_, i) => !(i % 5));
  return { n_inputs: 1, x_train: [train], y_train: train.map(f), x_val: [val], y_val: val.map(f),
           axes: [Array.from({ length: 200 }, (_, i) => -5 + i / 19.9)], total_steps: 0 };
}

const run = (cfg, meta = meta1d(), opts) => [...train({ ...BASE, ...cfg }, meta, opts)];

test('the loss goes down and the events have the trainer shapes', () => {
  const events = run({});
  const frames = events.filter(e => e.type === 'frame'), end = events.at(-1);
  assert.equal(events[0].type, 'start');
  assert.deepEqual(frames.map(f => f.step).slice(0, 3), [0, 1, 2]);
  assert.equal(frames[0].pred.length, 200);
  assert.equal(end.type, 'end');
  assert.equal(end.status, 'done');
  assert.equal(end.steps, 200);
  assert.ok(end.final_val < frames[0].val / 10, `${frames[0].val} → ${end.final_val}`);
  const steps = events.filter(e => e.type === 'loss').flatMap(e => e.steps);
  assert.deepEqual(steps, Array.from({ length: 200 }, (_, i) => i + 1));
});

test('every optimizer, loss and activation learns', () => {
  for (const optimizer of ['bgd', 'sgd', 'mini', 'momentum', 'rmsprop', 'adam']) {
    for (const [loss, activation] of [['mse', 'relu'], ['mae', 'gelu'], ['huber', 'elu'], ['mse', 'leaky_relu'], ['mse', 'sigmoid']]) {
      const lr = { bgd: 0.3, sgd: 0.01, mini: 0.1, momentum: 0.03, rmsprop: 0.005, adam: 0.02 }[optimizer];
      const events = run({ optimizer, loss, activation, learning_rate: lr, epochs: 300,
                           batch_method: ['bgd', 'sgd', 'mini'].includes(optimizer) ? optimizer : 'mini' });
      const first = events.find(e => e.type === 'frame').val, end = events.at(-1);
      assert.ok(end.final_val < first / 2, `${optimizer} ${loss} ${activation}: ${first} → ${end.final_val}`);
    }
  }
});

test('linear regression fits a line', () => {
  const end = run({ model: 'linear', learning_rate: 0.1 }, meta1d(x => 2 * x + 1)).at(-1);
  assert.ok(end.final_val < 1e-4, end.final_val);
});

test('the same seed gives the same run', () => {
  assert.deepEqual(run({}).at(-1).final_val, run({}).at(-1).final_val);
  assert.notEqual(run({}).at(-1).final_val, run({ seed: 7 }).at(-1).final_val);
});

test('a too large learning rate diverges', () => {
  const end = run({ optimizer: 'sgd', batch_method: 'sgd', learning_rate: 1, hidden_layers: [32, 32] }).at(-1);
  assert.equal(end.status, 'diverged');
  assert.equal(end.final_val, null);
});

test('the time limit stops a run', () => {
  let clock = 0;
  const end = run({ epochs: 50 }, meta1d(), { timeLimitS: 1, now: () => (clock += 100) }).at(-1);
  assert.equal(end.status, 'timeout');
  assert.ok(end.steps < 50);
});

test('two inputs: predictions on the 30 × 30 grid, x1 along each row', () => {
  const ax = Array.from({ length: 30 }, (_, i) => i - 15), pts = [];
  for (let i = 0; i < 200; i++) pts.push([((i * 7) % 30) - 15, ((i * 13) % 30) - 15]);
  const cols = p => [p.map(q => q[0]), p.map(q => q[1])], f = ([a]) => a;   // y = x1
  const meta = { n_inputs: 2, x_train: cols(pts.slice(40)), y_train: pts.slice(40).map(f),
                 x_val: cols(pts.slice(0, 40)), y_val: pts.slice(0, 40).map(f), axes: [ax, ax] };
  const events = run({ n_inputs: 2, model: 'linear', learning_rate: 0.1 }, meta);
  const pred = events.filter(e => e.type === 'frame').at(-1).pred;
  assert.equal(pred.length, 900);
  assert.ok(Math.abs(pred[29] - 14) < 0.1 && Math.abs(pred[30] - -15) < 0.1, `${pred[29]} ${pred[30]}`);
});
