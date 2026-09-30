// Custom inputs: hidden-layer editor, CSV upload with column pickers, and the data preview.
import { t } from './i18n.js';
import { api } from './api.js';
import { drawPrediction, fitCanvas, screenTheme } from './plot.js';

// ---------- ② hidden layers: [8, 8] ----------

export function mountLayersEditor(root, settings, { maxLayers, maxNeurons }) {
  let shown = '';
  const commit = layers => settings.set({ hidden_layers: layers });
  const clamp = n => Math.max(1, Math.min(maxNeurons, Math.round(+n || 1)));

  function render() {
    const layers = settings.values.hidden_layers || [];
    if (JSON.stringify(layers) === shown) return;
    shown = JSON.stringify(layers);
    root.replaceChildren();
    layers.forEach((n, i) => {
      const L = {
        name: t('Hidden {i}', { i: i + 1 }), dec: t('Fewer neurons'), inc: t('More neurons'),
        input: t('Neurons in hidden layer {i}', { i: i + 1 }), neurons: t('neurons'), del: t('Remove layer'),
      };
      const row = document.createElement('div');
      row.className = 'layer-row';
      row.innerHTML = `
        <span class="layer-name">${L.name}</span>
        <button type="button" class="btn btn-icon" data-act="dec" aria-label="${L.dec}">−</button>
        <input type="number" min="1" max="${maxNeurons}" value="${n}" aria-label="${L.input}">
        <button type="button" class="btn btn-icon" data-act="inc" aria-label="${L.inc}">+</button>
        <span class="muted">${L.neurons}</span>
        <button type="button" class="btn btn-icon btn-ghost" data-act="del" aria-label="${L.del}" ${layers.length <= 1 ? 'disabled' : ''}>✕</button>`;
      const input = row.querySelector('input');
      const update = v => { const next = [...settings.values.hidden_layers]; next[i] = clamp(v); commit(next); };
      row.querySelector('[data-act=dec]').onclick = () => update(n - 1);
      row.querySelector('[data-act=inc]').onclick = () => update(n + 1);
      row.querySelector('[data-act=del]').onclick = () => commit(settings.values.hidden_layers.filter((_, k) => k !== i));
      input.addEventListener('change', () => update(input.value));
      input.addEventListener('input', e => e.stopPropagation());
      root.append(row);
    });
    const add = document.createElement('button');
    add.type = 'button';
    add.className = 'btn btn-ghost btn-sm';
    add.textContent = t('+ Add layer');
    add.disabled = layers.length >= maxLayers;
    add.onclick = () => commit([...settings.values.hidden_layers, settings.values.hidden_layers.at(-1) || 8]);
    root.append(add);
  }

  settings.addEventListener('change', render);
  for (const b of document.querySelectorAll('[data-layers]')) {
    b.addEventListener('click', () => commit(JSON.parse(b.dataset.layers)));
  }
  render();
}

// ---------- ① CSV upload ----------

function decode(buffer) {
  try { return new TextDecoder('utf-8', { fatal: true }).decode(buffer).replace(/^﻿/, ''); }
  catch { return new TextDecoder('euc-kr').decode(buffer); }   // Excel in Korea often saves CP949
}

function headerOf(text) {
  const line = (text.split(/\r?\n/).find(l => l.trim()) || '');
  return line.split(',').map(c => c.trim().replace(/^"(.*)"$/, '$1')).filter(Boolean);
}

export function mountCsvInput(root, settings, { maxBytes, toast }) {
  const file = root.querySelector('[data-role=csv-file]');
  const features = root.querySelector('[data-role=csv-features]');
  const target = root.querySelector('select[name=csv_target]');
  const name = root.querySelector('[data-role=csv-name]');
  let shown = null;

  file.addEventListener('change', async () => {
    const f = file.files[0];
    if (!f) return;
    if (f.size > maxBytes) { toast(t('The CSV file is too large (max 1 MB).'), 'error'); file.value = ''; return; }
    const text = decode(await f.arrayBuffer());
    const cols = headerOf(text);
    if (cols.length < 2) { toast(t('The CSV needs a header row with at least two columns.'), 'error'); return; }
    settings.set({
      csv_name: f.name, csv_text: text, data_source: 'csv',
      csv_features: cols.slice(0, Math.min(2, cols.length - 1)), csv_target: cols.at(-1),
    });
    file.value = '';
  });

  features.addEventListener('change', e => {
    const checked = [...features.querySelectorAll('input:checked')];
    if (checked.length > 2) { e.target.checked = false; toast(t('Choose at most two input columns.'), 'error'); }
    else if (!checked.length) { e.target.checked = true; }
    settings.set({ csv_features: [...features.querySelectorAll('input:checked')].map(i => i.value) });
  });

  function render() {
    const v = settings.values;
    name.textContent = v.csv_name || t('No file yet');
    const cols = v.csv_text ? headerOf(v.csv_text) : [];
    if (JSON.stringify(cols) !== shown) {
      shown = JSON.stringify(cols);
      features.replaceChildren(...cols.map(c => {
        const label = document.createElement('label');
        label.className = 'chip';
        label.innerHTML = '<input type="checkbox"> <span></span>';
        label.querySelector('input').value = c;
        label.querySelector('span').textContent = c;
        return label;
      }));
      target.replaceChildren(...cols.map(c => new Option(c, c)));
    }
    for (const box of features.querySelectorAll('input')) box.checked = (v.csv_features || []).includes(box.value);
    target.value = v.csv_target;
  }

  settings.addEventListener('change', render);
  render();
}

// ---------- ① data preview (server builds the same data the trainer will use) ----------

const DATA_KEYS = ['data_source', 'n_inputs', 'function_1d', 'function_2d', 'expression_1d', 'expression_2d',
  'x_min', 'x_max', 'data_size', 'noise_std', 'validation_ratio', 'seed', 'csv_text', 'csv_features', 'csv_target'];

export function mountPreview(canvas, message, settings) {
  let last = '', timer = null, meta = null, seq = 0;

  const draw = () => {
    const { ctx, w, h } = fitCanvas(canvas);
    const th = screenTheme();
    ctx.fillStyle = th.bg;
    ctx.fillRect(0, 0, w, h);
    if (!meta) return;
    const cfg = settings.get();
    drawPrediction(ctx, { x: 0, y: 0, w, h }, { meta, config: cfg, frames: [] }, 0, th, 0.85, { badges: false });
  };

  async function fetchPreview() {
    const cfg = settings.get();
    const key = JSON.stringify(DATA_KEYS.map(k => cfg[k]));
    if (key === last) return;
    last = key;
    const mine = ++seq;
    try {
      const m = await api.preview(cfg);
      if (mine !== seq) return;
      meta = m;
      message.textContent = '';
      message.hidden = true;
    } catch (err) {
      if (mine !== seq) return;
      meta = null;
      message.textContent = err.messages ? err.messages.join(' ') : String(err);
      message.hidden = false;
    }
    draw();
  }

  settings.addEventListener('change', e => {
    if (!e.detail.keys.some(k => DATA_KEYS.includes(k))) return;
    clearTimeout(timer);
    timer = setTimeout(fetchPreview, 350);
  });
  new ResizeObserver(draw).observe(canvas);
  fetchPreview();
}
