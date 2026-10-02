// node --test tests/js   (history grouping, ★ per data, filters in app/static/js/history.js)
import { test } from 'node:test';
import assert from 'node:assert/strict';

globalThis.window = { OPTIONS: { presets: { 1: { abs: 'abs(x)', sin: 'sin(x)' }, 2: {} } } };
globalThis.document = { documentElement: { lang: 'ko' } };
const { dataKey } = await import('../../app/static/js/describe.js');
const { History } = await import('../../app/static/js/history.js');

const FN = { data_source: 'function', n_inputs: 1, function_1d: 'abs', expression_1d: '', data_size: 500, noise_std: 0.1,
  x_min: -5, x_max: 5, validation_ratio: 0.2, seed: 42, model: 'linear', loss: 'mse', optimizer: 'adam',
  learning_rate: 0.01, epochs: 100, batch_method: 'mini', batch_size: 32, hidden_layers: [4], activation: 'relu' };

let id = 0;
const row = (cfg, final_val, created_at = '2026-10-02T01:00:00Z') =>
  ({ id: ++id, seq: id, created_at, status: 'done', final_train: final_val, final_val, steps: 1, duration: 1, config: { ...FN, ...cfg } });

function table(rows) {
  const el = { innerHTML: '', addEventListener() {} };
  const h = new History(el, {});
  h.setRows(rows);
  return { h, el, stars: () => (el.innerHTML.match(/class="star"/g) || []).length, shown: () => h.visible.map(r => r.id) };
}

test('the same formula is the same data, whatever the points, noise or seed', () => {
  assert.equal(dataKey({ ...FN }), dataKey({ ...FN, data_size: 50, noise_std: 1, seed: 7, x_min: -1 }));
  assert.equal(dataKey({ ...FN }), dataKey({ ...FN, function_1d: 'custom', expression_1d: 'abs(x)' }));
  assert.notEqual(dataKey({ ...FN }), dataKey({ ...FN, function_1d: 'sin' }));
  const csv = { data_source: 'csv', csv_name: 'a.csv', csv_features: ['x'], csv_target: 'y' };
  assert.equal(dataKey({ ...FN, ...csv }), dataKey({ ...FN, ...csv, seed: 1 }));
  assert.notEqual(dataKey({ ...FN, ...csv }), dataKey({ ...FN, ...csv, csv_target: 'z' }));
});

test('every data set gets a star on its lowest validation loss', () => {
  const rows = [row({}, 0.5), row({ noise_std: 1 }, 0.2), row({ function_1d: 'sin' }, 0.9), row({ function_1d: 'sin' }, 0.4)];
  const { h, stars } = table(rows.reverse());
  assert.equal(stars(), 2);
  const best = [...h.table.innerHTML.matchAll(/<tr data-id="(\d+)"[\s\S]*?<\/tr>/g)]
    .filter(m => m[0].includes('class="star"')).map(m => +m[1]).sort();
  assert.deepEqual(best, [rows.find(r => r.final_val === 0.2).id, rows.find(r => r.final_val === 0.4).id].sort());
});

test('the data and date filters limit the rows; a deleted data set clears its filter', () => {
  const a = row({}, 0.5, '2026-09-30T01:00:00Z'), b = row({ function_1d: 'sin' }, 0.4), c = row({}, 0.3);
  const { h, shown } = table([c, b, a]);
  assert.deepEqual([...h.dataSets().values()].map(s => s.count), [2, 1]);
  h.setFilter({ data: dataKey(a.config) });
  assert.deepEqual(shown(), [c.id, a.id]);
  h.setFilter({ from: '2026-10-02', to: '2026-10-02' });
  assert.deepEqual(shown(), [c.id]);
  h.setFilter({ from: '', to: '', data: dataKey(b.config) });
  h.remove(b.id);
  assert.equal(h.filter.data, '');
});

test('the last loaded row is compared with the run just below the page', () => {
  const older = row({ learning_rate: 0.1 }, 0.5);
  const page = [row({}, 0.4), row({}, 0.3)].reverse();
  const el = { innerHTML: '', addEventListener() {} };
  const h = new History(el, {});
  h.setRows(page, { older, total: 3 });
  assert.equal(h.previous(page[1]), older);
  assert.equal((el.innerHTML.match(/class="changed"/g) || []).length, 1);
  assert.ok(!el.innerHTML.includes(`data-id="${older.id}"`));    // shown only after it is loaded
  h.addOlder([older], null, 3);
  assert.equal(h.previous(older), null);
  h.remove(page[0].id);
  assert.equal(h.total, 2);
});

test('sorting by validation loss: lowest first, highest first, then newest first again', () => {
  const rows = [row({}, 0.5), row({}, null), row({}, 0.2), row({}, 0.9)].reverse();
  const { h, shown } = table(rows);
  const ids = vals => vals.map(v => rows.find(r => r.final_val === v).id);
  h.setSort('asc');
  assert.deepEqual(shown(), ids([0.2, 0.5, 0.9, null]));
  h.setSort('desc');
  assert.deepEqual(shown(), ids([0.9, 0.5, 0.2, null]));
  h.setSort('');
  assert.deepEqual(shown(), rows.map(r => r.id));
  h.setFilter({ data: dataKey(FN) });
  h.setSort('asc');
  assert.equal(h.visible.length, 4);
  assert.equal(h.previous(rows[0]), rows[1]);          // "changed from the run before" stays by time
});
