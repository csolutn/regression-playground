// Human-readable names and one-line summaries of settings.
// Used by the step cards, the history table, the plot badges and the MP4 title.
import { t } from './i18n.js';

const OPTIONS = () => window.OPTIONS || { presets: { 1: {}, 2: {} } };
export const GRADIENT_MODELS = ['linear', 'neural_net'];
export const TREE_MODELS = ['decision_tree', 'random_forest', 'gradient_boosting'];
export const isTree = cfg => TREE_MODELS.includes(cfg.model);
export const PLAIN_GD = ['bgd', 'sgd', 'mini'];     // the optimizer's name fixes the batch (app/ml/options.py)

export const modelName = m => ({
  linear: t('Linear regression'), neural_net: t('Neural network'), decision_tree: t('Decision tree'),
  random_forest: t('Random forest'), gradient_boosting: t('Gradient boosting'),
})[m] || m;

export const lossName = cfg => ({ mse: 'MSE', mae: 'MAE (L1)', huber: `Huber (δ=${cfg.huber_delta})` })[cfg.loss];
export const optimizerName = o => ({
  bgd: 'BGD', mini: t('Mini-batch'), sgd: 'SGD', momentum: 'Momentum', rmsprop: 'RMSprop', adam: 'Adam',
})[o] || o;
export const activationName = a => ({
  relu: 'ReLU', leaky_relu: 'Leaky ReLU', elu: 'ELU', gelu: 'GELU', tanh: 'tanh', sigmoid: 'sigmoid', linear: t('none (linear)'),
})[a] || a;

export function stepLabel(model) {
  return ({ decision_tree: t('Depth'), random_forest: t('Trees'), gradient_boosting: t('Boosting step') })[model] || t('Epoch');
}

export function structure(cfg) {
  return [cfg.n_inputs, ...(cfg.model === 'neural_net' ? cfg.hidden_layers : []), 1];
}

export function formula(cfg) {
  const n = String(cfg.n_inputs);
  const key = n === '1' ? cfg.function_1d : cfg.function_2d;
  if (key === 'custom') return n === '1' ? cfg.expression_1d : cfg.expression_2d;
  return OPTIONS().presets[n]?.[key] ?? key;
}

// abs(x) → |x|, x**2 → x², 2*x → 2·x
export function prettyFormula(f) {
  return String(f)
    .replace(/abs\(([^()]*)\)/g, '|$1|')
    .replace(/\*\*2\b/g, '²').replace(/\*\*3\b/g, '³')
    .replace(/\s*\*\s*/g, '·');
}

export function targetLabel(cfg) {
  if (cfg.data_source === 'csv') return `${cfg.csv_target || '?'} (${(cfg.csv_features || []).join(', ')})`;
  return prettyFormula(formula(cfg));
}

export function dataSummary(cfg) {
  if (cfg.data_source === 'csv') return `${t('CSV')} ${cfg.csv_name || ''} · ${targetLabel(cfg)}`;
  const points = t('{n} points', { n: cfg.data_size });
  return `y = ${prettyFormula(formula(cfg))} · ${points} · ${t('noise')} ${cfg.noise_std}`;
}

export function modelSummary(cfg) {
  if (cfg.model === 'linear') return `${modelName('linear')} [${structure(cfg).join(', ')}]`;
  if (cfg.model === 'neural_net') return `${modelName('neural_net')} [${structure(cfg).join(', ')}] · ${activationName(cfg.activation)}`;
  const depth = t('depth {d}', { d: cfg.max_depth });
  if (cfg.model === 'decision_tree') return `${modelName(cfg.model)} · ${depth}`;
  return `${modelName(cfg.model)} · ${depth} × ${cfg.n_estimators}`;
}

export const lossSummary = cfg => lossName(cfg);

export function optimSummary(cfg) {
  if (cfg.model === 'gradient_boosting') return t('Boosting lr {lr}', { lr: cfg.tree_learning_rate });
  if (isTree(cfg)) return t('Splitting (no gradient descent)');
  const batch = { sgd: t('batch 1'), mini: t('batch {n}', { n: cfg.batch_size }), bgd: t('full batch') }[cfg.batch_method];
  const epochs = t('{n} epochs', { n: cfg.epochs });
  return `${optimizerName(cfg.optimizer)} · lr ${cfg.learning_rate} · ${epochs} · ${batch}`;
}

// Lines in the top-right badge of the prediction plot (like the reference video)
export function configBadge(cfg) {
  const loss = lossName(cfg);
  if (cfg.model === 'neural_net') {
    return [`${t('Activation')}: ${activationName(cfg.activation)} • ${loss}`, `${t('Widths')}: [${structure(cfg).join(', ')}]`];
  }
  if (cfg.model === 'linear') {
    return [`${modelName('linear')} • ${loss}`, `${optimizerName(cfg.optimizer)} • lr ${cfg.learning_rate}`];
  }
  return [`${modelName(cfg.model)} • ${loss}`, modelSummary(cfg).split(' · ').slice(1).join(' · ')];
}

// Default MP4 title, e.g. "Can a ReLU neural network learn x·sin(x)?"
export function videoTitle(cfg) {
  const f = targetLabel(cfg);
  if (cfg.model === 'neural_net') return t('Can a {act} neural network learn {f}?', { act: activationName(cfg.activation), f });
  if (cfg.model === 'linear') return t('Can linear regression learn {f}?', { f });
  return t('Can a {model} learn {f}?', { model: modelName(cfg.model).toLowerCase(), f });
}
