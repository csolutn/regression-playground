// Playground page: settings on the left (① → ④), output on the right.
import { t } from './i18n.js';
import { api } from './api.js';
import { Settings } from './settings.js';
import { OutputPanel } from './output.js';
import { mountCsvInput, mountLayersEditor, mountPreview } from './widgets.js';
import { renderActivation, renderLossShape, renderStructure } from './structure.js';
import { dataSummary, isTree, lossSummary, modelSummary, optimSummary } from './describe.js';
import { toast } from './toast.js';

const OPTIONS = window.OPTIONS;
const root = document.querySelector('[data-page=playground]');
const form = document.getElementById('settings');
const settings = new Settings(form, OPTIONS.defaults, `mlp.settings.${root.dataset.user}`);

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
mountLayersEditor(document.getElementById('layers-editor'), settings,
  { maxLayers: OPTIONS.max_hidden_layers, maxNeurons: OPTIONS.max_neurons });
mountCsvInput(document.getElementById('csv-input'), settings, { maxBytes: OPTIONS.max_csv_bytes, toast });
mountPreview(document.getElementById('data-preview'), document.getElementById('preview-msg'), settings);
refresh();

document.getElementById('reset-btn').addEventListener('click', () => {
  if (confirm(t('Reset all settings to the defaults?'))) settings.reset();
});

// ---------- output and training ----------

const output = new OutputPanel(document.getElementById('output'), {
  toast,
  onLoadSettings: run => {
    settings.set(run.config);
    toast(t('Loaded the settings of run #{n}.', { n: run.row.seq }), 'info');
  },
});
output.init();

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
    await api.train(cfg, ev => output.liveEvent(run, ev), controller.signal);
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
