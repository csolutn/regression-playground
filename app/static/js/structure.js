// Small SVG diagrams in the settings panels:
// ② model structure (network, or "tree × count"), activation shape, ③ loss shape.
import { t } from './i18n.js';
import { activationName, isTree, structure } from './describe.js';

const NS = 'http://www.w3.org/2000/svg';
const MAX_NEURONS_DRAWN = 8;
const MAX_TREE_DEPTH_DRAWN = 5;

function el(name, attrs = {}, text) {
  const e = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}

// ---------- ② model structure ----------

export function renderStructure(svg, caption, cfg) {
  svg.replaceChildren();
  if (isTree(cfg)) drawForest(svg, caption, cfg);
  else drawNetwork(svg, caption, cfg);
}

function drawNetwork(svg, caption, cfg) {
  const W = 520, H = 230;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const sizes = structure(cfg);
  const xs = sizes.map((_, i) => 60 + i * (W - 120) / Math.max(1, sizes.length - 1));
  const ys = sizes.map(n => {
    const k = Math.min(n, MAX_NEURONS_DRAWN);
    return k === 1 ? [H / 2] : Array.from({ length: k }, (_, j) => 48 + j * (H - 96) / (k - 1));
  });
  const edges = el('g', { class: 'edges' });
  for (let i = 0; i < sizes.length - 1; i++) {
    const opacity = Math.max(0.12, Math.min(0.6, 3 / Math.sqrt(ys[i].length * ys[i + 1].length)));
    for (const y1 of ys[i]) for (const y2 of ys[i + 1]) {
      edges.append(el('line', { x1: xs[i], y1, x2: xs[i + 1], y2, 'stroke-opacity': opacity }));
    }
  }
  svg.append(edges);
  sizes.forEach((n, i) => {
    const kind = i === 0 ? 'input' : i === sizes.length - 1 ? 'output' : 'hidden';
    const g = el('g', { class: `layer ${kind}` });
    ys[i].forEach(y => g.append(el('circle', { cx: xs[i], cy: y, r: 9 })));
    if (n > MAX_NEURONS_DRAWN) g.append(el('text', { x: xs[i] + 16, y: H / 2 + 4, class: 'more' }, '⋮'));
    g.append(el('text', { x: xs[i], y: 22, class: 'count' }, String(n)));
    const name = kind === 'input' ? t('Input') : kind === 'output' ? t('Output') : t('Hidden {i}', { i });
    g.append(el('text', { x: xs[i], y: H - 12, class: 'name' }, name));
    svg.append(g);
  });
  if (cfg.model === 'neural_net') {
    for (let i = 1; i < sizes.length - 1; i++) {
      svg.append(el('text', { x: xs[i], y: H - 30, class: 'act' }, activationName(cfg.activation)));
    }
  }
  const params = sizes.slice(1).reduce((sum, n, i) => sum + n * sizes[i] + n, 0);
  if (cfg.model === 'linear') {
    const f = cfg.n_inputs === 1 ? 'ŷ = w·x + b' : 'ŷ = w₁·x₁ + w₂·x₂ + b';
    caption.textContent = t('{f}: a straight line (or flat plane), {n} numbers to learn.', { f, n: params });
  } else {
    caption.textContent = t('{n} weights and biases to learn. Each hidden neuron bends the line with {act}.',
      { n: params.toLocaleString(), act: activationName(cfg.activation) });
  }
}

function drawForest(svg, caption, cfg) {
  const W = 520, H = 230;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const depth = cfg.max_depth, shown = Math.min(depth, MAX_TREE_DEPTH_DRAWN);
  const count = cfg.model === 'decision_tree' ? 1 : cfg.n_estimators;
  const box = { x: 30, y: 30, w: count > 1 ? 330 : 440, h: 150 };

  // a stack of cards behind the tree when there are many trees
  for (let k = Math.min(count, 3) - 1; k >= 1; k--) {
    svg.append(el('rect', { x: box.x - 12 + k * 10, y: box.y - 16 - k * 8, width: box.w + 24, height: box.h + 36, rx: 10, class: 'tree-card ghost' }));
  }
  svg.append(el('rect', { x: box.x - 12, y: box.y - 16, width: box.w + 24, height: box.h + 36, rx: 10, class: 'tree-card' }));

  const pos = (level, j) => [box.x + (j + 0.5) / 2 ** level * box.w, box.y + (shown ? level / shown : 0) * box.h];
  const g = el('g', { class: 'tree' });
  for (let level = 0; level < shown; level++) {
    for (let j = 0; j < 2 ** level; j++) {
      const [x1, y1] = pos(level, j);
      for (const c of [2 * j, 2 * j + 1]) {
        const [x2, y2] = pos(level + 1, c);
        g.append(el('line', { x1, y1, x2, y2 }));
      }
    }
  }
  const leafR = Math.max(2, Math.min(7, box.w / 2 ** shown / 3));
  for (let level = 0; level <= shown; level++) {
    for (let j = 0; j < 2 ** level; j++) {
      const [x, y] = pos(level, j);
      if (level < shown) g.append(el('rect', { x: x - 5, y: y - 5, width: 10, height: 10, class: 'node-split' }));
      else g.append(el('circle', { cx: x, cy: y, r: leafR, class: 'leaf' }));
    }
  }
  svg.append(g);
  const label = depth > shown ? t('depth {d} (first {k} levels shown)', { d: depth, k: shown }) : t('depth {d}', { d: depth });
  svg.append(el('text', { x: box.x + box.w / 2, y: H - 6, class: 'name' }, label));
  if (depth > shown) svg.append(el('text', { x: box.x + box.w / 2, y: box.y + box.h + 16, class: 'more' }, '⋯'));
  if (count > 1) {
    svg.append(el('text', { x: 440, y: 112, class: 'times' }, `× ${count}`));
    svg.append(el('text', { x: 440, y: 140, class: 'name' },
      cfg.model === 'random_forest' ? t('average') : t('sum in order')));
  }

  const leaves = 2 ** depth;
  caption.textContent = {
    decision_tree: t('One tree of depth {d}: up to {leaves} boxes (leaves), each predicting one value. The animation grows the depth from 1 to {d}.', { d: depth, leaves }),
    random_forest: t('{n} trees of depth {d}, each trained on a random sample of the data. The prediction is their average.', { n: count, d: depth }),
    gradient_boosting: t('{n} trees of depth {d}, added one by one. Each new tree corrects the error that is left.', { n: count, d: depth }),
  }[cfg.model];
}

// ---------- mini function plots ----------

const ACTIVATIONS = {
  relu: x => Math.max(0, x),
  leaky_relu: x => (x > 0 ? x : 0.01 * x),
  elu: x => (x > 0 ? x : Math.exp(x) - 1),
  gelu: x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))),
  tanh: Math.tanh,
  sigmoid: x => 1 / (1 + Math.exp(-x)),
  linear: x => x,
};

function miniPlot(svg, curves, { xr, yr, W = 220, H = 120 }) {
  svg.replaceChildren();
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const X = x => 8 + (x - xr[0]) / (xr[1] - xr[0]) * (W - 16);
  const Y = y => H - 8 - (y - yr[0]) / (yr[1] - yr[0]) * (H - 16);
  svg.append(el('line', { x1: X(xr[0]), y1: Y(0), x2: X(xr[1]), y2: Y(0), class: 'axis' }));
  svg.append(el('line', { x1: X(0), y1: Y(yr[0]), x2: X(0), y2: Y(yr[1]), class: 'axis' }));
  for (const c of curves) {
    const pts = [];
    for (let i = 0; i <= 120; i++) {
      const x = xr[0] + (xr[1] - xr[0]) * i / 120;
      const y = Math.max(yr[0] - 1, Math.min(yr[1] + 1, c.f(x)));
      pts.push(`${X(x).toFixed(1)},${Y(y).toFixed(1)}`);
    }
    svg.append(el('polyline', { points: pts.join(' '), class: c.active ? 'curve active' : 'curve' }));
    if (c.label) {
      const lx = c.lx ?? xr[1] * 0.72, ly = Math.min(yr[1] - 0.3, c.f(lx));
      svg.append(el('text', { x: X(lx), y: Y(ly) - 6, class: c.active ? 'curve-label active' : 'curve-label' }, c.label));
    }
  }
}

export function renderActivation(svg, name) {
  miniPlot(svg, [{ f: ACTIVATIONS[name] || ACTIVATIONS.linear, active: true }], { xr: [-3, 3], yr: [-1.5, 3] });
}

export function renderLossShape(svg, cfg) {
  const d = +cfg.huber_delta || 1;
  const curves = [
    { key: 'mse', f: e => e * e, label: 'MSE', lx: 1.45 },
    { key: 'mae', f: e => Math.abs(e), label: 'MAE', lx: -2.5 },
    { key: 'huber', f: e => (Math.abs(e) <= d ? 0.5 * e * e : d * (Math.abs(e) - 0.5 * d)), label: 'Huber', lx: 2.6 },
  ].map(c => ({ ...c, active: c.key === cfg.loss }));
  curves.sort((a, b) => a.active - b.active);   // selected curve on top
  miniPlot(svg, curves, { xr: [-3, 3], yr: [0, 4.5], W: 260, H: 140 });
}
