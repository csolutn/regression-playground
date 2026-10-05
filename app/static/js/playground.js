// Playground page: settings on the left (① → ④), output on the right.
import { t } from './i18n.js';
import { api } from './api.js';
import { trainRun } from './train.js';
import { Settings } from './settings.js';
import { OutputPanel } from './output.js';
import { mountCsvInput, mountLayersEditor } from './widgets.js';
import { renderActivation, renderLossShape, renderStructure } from './structure.js';
import { PLAIN_GD, dataSummary, isTree, lossSummary, modelSummary, optimSummary } from './describe.js';
import { toast } from './toast.js';

const OPTIONS = window.OPTIONS;
const root = document.querySelector('[data-page=playground]');
const form = document.getElementById('settings');
const guest = root.dataset.guest === '1';        // no database: runs train in this browser and stay in this page
const settings = new Settings(form, OPTIONS.defaults, `mlp.settings.v3.${root.dataset.user}`,
  guest ? () => sessionStorage : () => localStorage);

// ---------- step cards ① → ④ ----------

const SUMMARIES = { data: dataSummary, model: modelSummary, loss: lossSummary, optim: optimSummary };
const STEP_KEY = `mlp.step.${root.dataset.user}`;

function showStep(step) {
  for (const card of form.querySelectorAll('.step-card')) {
    const on = card.dataset.step === step;
    card.classList.toggle('active', on);
    card.setAttribute('aria-selected', String(on));
  }
  for (const panel of form.querySelectorAll('.step-panel')) panel.hidden = panel.dataset.panel !== step;
  try { localStorage.setItem(STEP_KEY, step); } catch { /* ignore */ }
}

for (const card of form.querySelectorAll('.step-card')) card.addEventListener('click', () => showStep(card.dataset.step));
let initialStep = 'data';
try { initialStep = localStorage.getItem(STEP_KEY) || 'data'; } catch { /* ignore */ }
showStep(initialStep);

// ---------- things that follow the settings ----------

const structureSvg = document.getElementById('structure-svg');
const structureCaption = document.getElementById('structure-caption');
const activationSvg = document.getElementById('activation-svg');
const lossSvg = document.getElementById('loss-svg');
const modelAdvanced = document.getElementById('model-advanced');

function refresh() {
  const cfg = settings.get();
  for (const [step, fn] of Object.entries(SUMMARIES)) {
    form.querySelector(`[data-summary="${step}"]`).textContent = fn(cfg);
  }
  renderStructure(structureSvg, structureCaption, cfg);
  renderActivation(activationSvg, cfg.activation);
  renderLossShape(lossSvg, cfg);
  if (isTree(cfg)) modelAdvanced.open = true;
}
settings.addEventListener('change', refresh);

// BGD / Mini-batch / SGD fix the batch; switching from one of them to Momentum, RMSprop or Adam starts at mini-batch
let lastOptimizer = settings.values.optimizer;
function syncBatch() {
  const { optimizer, batch_method } = settings.values, prev = lastOptimizer;
  lastOptimizer = optimizer;
  if (PLAIN_GD.includes(optimizer)) {
    if (batch_method !== optimizer) settings.set({ batch_method: optimizer });
  } else if (optimizer !== prev && PLAIN_GD.includes(prev)) settings.set({ batch_method: 'mini' });
}
settings.addEventListener('change', syncBatch);
syncBatch();
mountLayersEditor(document.getElementById('layers-editor'), settings,
  { maxLayers: OPTIONS.max_hidden_layers, maxNeurons: OPTIONS.max_neurons });
mountCsvInput(document.getElementById('csv-input'), settings, { maxBytes: OPTIONS.max_csv_bytes, toast });
refresh();

// ---------- output and training ----------

const output = new OutputPanel(document.getElementById('output'), {
  toast,
  guest,
  onLoadSettings: run => {
    settings.set(run.config);
    dataPreview.cancel();             // keep showing that run, not its data
    toast(t('Loaded the settings of run #{n}.', { n: run.row.seq }), 'info');
  },
});

// Changing ① training data (or resetting) shows the new data in the output, as points only,
// before training. The server builds the same data the trainer will use.
const DATA_KEYS = ['data_source', 'n_inputs', 'function_1d', 'function_2d', 'expression_1d', 'expression_2d',
  'x_min', 'x_max', 'data_size', 'noise_std', 'validation_ratio', 'seed', 'csv_text', 'csv_features', 'csv_target'];
const dataPreview = {
  key: '', timer: null, seq: 0,
  watch() {
    this.key = this.keyOf(settings.values);
    settings.addEventListener('change', () => {
      const key = this.keyOf(settings.values);
      if (key === this.key) return;
      this.key = key;
      clearTimeout(this.timer);
      this.timer = setTimeout(() => this.show(), 350);
    });
  },
  keyOf: cfg => JSON.stringify(DATA_KEYS.map(k => cfg[k])),
  cancel() { clearTimeout(this.timer); this.seq++; },
  async show() {
    this.cancel();
    if (output.run?.live) return;                 // never replace a run that is training
    const mine = this.seq, shown = output.run, cfg = settings.get();
    let meta = null, error = null;
    try { meta = await api.preview(cfg); }
    catch (err) { error = err.messages ? err.messages.join(' ') : err.message; }
    if (mine === this.seq && output.run === shown) output.showPreview(cfg, meta, error);   // nothing else shown meanwhile
  },
};
dataPreview.watch();
output.init().then(() => { if (!output.run) dataPreview.show(); });    // no runs yet: show the data

document.getElementById('reset-btn').addEventListener('click', () => {
  if (!confirm(t('Reset all settings to the defaults?'))) return;
  settings.reset();
  showStep('data');
  dataPreview.show();
});

const trainBtn = document.getElementById('train-btn');
let controller = null;

async function train() {
  if (controller) { controller.abort(); return; }
  const cfg = settings.get();
  controller = new AbortController();
  trainBtn.classList.add('is-running');
  trainBtn.textContent = t('■ Stop');
  const run = output.startLive(cfg);
  try {
    await trainRun(cfg, ev => output.liveEvent(run, ev), controller.signal, { guest });
    if (run.live) output.stopLive(run, t('The connection closed before the run was saved.'));
  } catch (err) {
    if (err.name === 'AbortError') output.stopLive(run, t('Stopped (not saved)'));
    else {
      output.stopLive(run, t('Error'));
      toast(err.messages ? err.messages.join('\n') : err.message, 'error');
    }
  } finally {
    controller = null;
    trainBtn.classList.remove('is-running');
    trainBtn.textContent = t('▶ Train');
  }
}

trainBtn.addEventListener('click', train);
document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); if (!controller) train(); }
});
