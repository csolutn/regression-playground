// node --test tests/js   (the HTML backup trains a browser run again on its page: app/static/js/replay-here.js)
import { test } from 'node:test';
import assert from 'node:assert/strict';

globalThis.window = { OPTIONS: { presets: { 1: {}, 2: {} } } };
globalThis.document = { documentElement: { lang: 'ko' } };
const { train, TRAINER_VERSION } = await import('../../app/static/js/nn.js');
const { replayHere } = await import('../../app/static/js/replay-here.js');

const CFG = {
  n_inputs: 1, model: 'neural_net', hidden_layers: [8], activation: 'tanh', loss: 'mse', huber_delta: 1,
  optimizer: 'adam', learning_rate: 0.02, epochs: 400, batch_method: 'mini', batch_size: 16, seed: 42,
};
const xs = Array.from({ length: 120 }, (_, i) => -5 + (10 * i) / 119);
const tr = xs.filter((_, i) => i % 5), va = xs.filter((_, i) => !(i % 5));
const META = { n_inputs: 1, x_train: [tr], y_train: tr.map(Math.abs), x_val: [va], y_val: va.map(Math.abs),
               axes: [Array.from({ length: 200 }, (_, i) => -5 + i / 19.9)] };

function entry(steps, finalVal) {
  return { row: { id: 1, steps, final_val: finalVal, config: CFG }, meta: 0, total_steps: CFG.epochs,
           replay: { trainer: TRAINER_VERSION, data_changed: false } };
}

test('a replay on the page ends where the run ended, with its last frame and a thinned loss curve', async () => {
  const events = [...train({ ...CFG, epochs: 250 }, META)];         // the run stopped at epoch 250 (a time limit)
  const end = events.at(-1), last = events.filter(e => e.type === 'frame').at(-1);
  const seen = [];
  const r = await replayHere(entry(250, end.final_val), [META], p => seen.push(p), new AbortController().signal);
  assert.equal(r.frame.step, 250);
  assert.deepEqual(r.frame.pred, last.pred.map(v => Number(v.toPrecision(4))));
  assert.equal(r.loss.steps.at(-1), 250);
  assert.ok(r.loss.steps.length <= 201);
  assert.equal(r.note, undefined);                                   // the same numbers as saved
  assert.equal(seen.at(-1), 1);
});

test('it says when the replay may differ, and it stops when asked', async () => {
  const r = await replayHere(entry(50, 123), [META], () => {}, new AbortController().signal);
  assert.match(r.note, /differ/);
  const ac = new AbortController();
  ac.abort();
  const big = { ...entry(400, null), row: { id: 2, steps: 400, final_val: null, config: { ...CFG, hidden_layers: [64, 64] } } };
  await assert.rejects(replayHere(big, [META], () => {}, ac.signal), { name: 'AbortError' });
});
