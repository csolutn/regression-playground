// History table: newest run on top; a cell is highlighted when it differs from the run before it.
import { t } from './i18n.js';
import { fmtLoss } from './plot.js';
import { dataSummary, lossSummary, modelSummary, optimSummary } from './describe.js';

// The four setting columns follow the pipeline ① → ④; `get` turns a run's settings into text.
export const SETTING_COLUMNS = [
  { key: 'data', label: () => t('① Data'), get: dataSummary },
  { key: 'model', label: () => t('② Model'), get: modelSummary },
  { key: 'loss', label: () => t('③ Loss'), get: lossSummary },
  { key: 'optim', label: () => t('④ Optimization'), get: optimSummary },
];

const LANG = document.documentElement.lang || undefined;
const STATUS = () => ({ timeout: t('time limit'), diverged: t('diverged') });

export class History {
  constructor(table, { readonly = false, onView, onLoad, onDelete }) {
    this.table = table;
    this.readonly = readonly;
    this.handlers = { onView, onLoad, onDelete };
    this.rows = [];
    this.activeId = null;
    table.addEventListener('click', e => this.onClick(e));
  }

  setRows(rows) { this.rows = rows; this.render(); }
  add(row) { this.rows.unshift(row); this.render(); }
  remove(id) { this.rows = this.rows.filter(r => r.id !== id); this.render(); }
  setActive(id) { this.activeId = id; this.render(); }
  previous(row) { return this.rows[this.rows.indexOf(row) + 1] || null; }

  onClick(e) {
    const btn = e.target.closest('button[data-act]');
    const tr = e.target.closest('tr[data-id]');
    if (!tr) return;
    const row = this.rows.find(r => r.id === +tr.dataset.id);
    if (!btn) { this.handlers.onView?.(row); return; }
    const fn = { view: 'onView', load: 'onLoad', del: 'onDelete' }[btn.dataset.act];
    this.handlers[fn]?.(row);
  }

  render() {
    const best = this.rows.filter(r => r.final_val != null).reduce((b, r) => (!b || r.final_val < b.final_val ? r : b), null);
    const head = `<thead><tr>
      <th>#</th><th>${t('Time')}</th>
      ${SETTING_COLUMNS.map(c => `<th>${c.label()}</th>`).join('')}
      <th class="num">${t('Train loss')}</th><th class="num">${t('Validation loss')}</th><th class="num">${t('Seconds')}</th><th></th>
    </tr></thead>`;
    const L = { view: esc(t('Show this run')), load: esc(t('Load these settings')), del: esc(t('Delete')), best: esc(t('Lowest validation loss')) };
    const body = this.rows.map(r => {
      const prev = this.previous(r);
      const cells = SETTING_COLUMNS.map(c => {
        const now = c.get(r.config), before = prev ? c.get(prev.config) : now;
        if (now === before) return `<td>${esc(now)}</td>`;
        const title = esc(t('Before: {v}', { v: before }));
        return `<td class="changed" title="${title}">${esc(now)}</td>`;
      }).join('');
      const status = STATUS()[r.status];
      const time = new Date(r.created_at);
      return `<tr data-id="${r.id}" class="${r.id === this.activeId ? 'active' : ''}">
        <td class="seq">${r.seq}</td>
        <td class="time" title="${time.toLocaleString(LANG)}">${time.toLocaleTimeString(LANG, { hour: '2-digit', minute: '2-digit' })}</td>
        ${cells}
        <td class="num">${fmtLoss(r.final_train)}</td>
        <td class="num ${r === best ? 'best' : ''}">${fmtLoss(r.final_val)}${r === best ? ` <span class="star" title="${L.best}">★</span>` : ''}${status ? ` <span class="pill pill-warn">${status}</span>` : ''}</td>
        <td class="num">${r.duration.toFixed(1)}</td>
        <td class="actions">
          <button type="button" class="btn btn-icon btn-ghost" data-act="view" title="${L.view}">▶</button>
          ${this.readonly ? '' : `<button type="button" class="btn btn-icon btn-ghost" data-act="load" title="${L.load}">⟲</button>
          <button type="button" class="btn btn-icon btn-ghost" data-act="del" title="${L.del}">🗑</button>`}
        </td></tr>`;
    }).join('');
    const empty = `<tbody><tr><td colspan="10" class="empty">${t('No runs yet.')}</td></tr></tbody>`;
    this.table.innerHTML = head + (this.rows.length ? `<tbody>${body}</tbody>` : empty);
  }
}

// What changed from the previous run, and how the validation loss moved
export function diffSummary(row, prev) {
  if (!prev) return { changes: [], loss: null };
  const changes = SETTING_COLUMNS
    .map(c => ({ label: c.label(), ...changedParts(c.get(prev.config), c.get(row.config)) }))
    .filter(c => c.before !== c.now);
  let loss = null;
  if (row.final_val != null && prev.final_val != null && prev.final_val > 0) {
    loss = { before: prev.final_val, now: row.final_val, ratio: row.final_val / prev.final_val - 1 };
  }
  return { changes, loss, prevSeq: prev.seq };
}

// "Adam · lr 0.01 · 300 epochs" vs "Adam · lr 0.1 · 300 epochs" → only "lr 0.01" / "lr 0.1"
function changedParts(before, now) {
  const a = before.split(' · '), b = now.split(' · ');
  if (a.length !== b.length) return { before, now };
  const idx = a.map((_, i) => i).filter(i => a[i] !== b[i]);
  return { before: idx.map(i => a[i]).join(' · '), now: idx.map(i => b[i]).join(' · ') };
}

export function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
