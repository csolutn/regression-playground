// History table: newest run on top; a cell is highlighted when it differs from the run before it.
// A date filter limits the rows shown, and the shown rows can be saved as CSV.
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
    this.filter = { from: '', to: '' };      // local dates 'YYYY-MM-DD', '' = open
    this.onRender = null;
    table.addEventListener('click', e => this.onClick(e));
  }

  setRows(rows) { this.rows = rows; this.render(); }
  add(row) { this.rows.unshift(row); this.render(); }
  remove(id) { this.rows = this.rows.filter(r => r.id !== id); this.render(); }
  setActive(id) { this.activeId = id; this.render(); }
  setFilter(filter) { this.filter = { ...this.filter, ...filter }; this.render(); }
  // "before" is always the run just before, even when the filter hides it
  previous(row) { return this.rows[this.rows.indexOf(row) + 1] || null; }

  get filtered() { return !!(this.filter.from || this.filter.to); }
  get visible() {
    const { from, to } = this.filter;
    return this.rows.filter(r => { const d = localDate(r.created_at); return (!from || d >= from) && (!to || d <= to); });
  }

  // the shown rows as CSV (UTF-8 with BOM so Excel reads Korean)
  toCsv() {
    const head = ['#', t('Time'), ...SETTING_COLUMNS.map(c => c.label()), t('Train loss'), t('Validation loss'),
      t('Steps'), t('Seconds'), t('Status'), t('Settings (JSON)')];
    const lines = this.visible.map(r => [
      r.seq, localDateTime(r.created_at), ...SETTING_COLUMNS.map(c => c.get(r.config)),
      r.final_train, r.final_val, r.steps, r.duration, STATUS()[r.status] || '', JSON.stringify(r.config),
    ]);
    return '\ufeff' + [head, ...lines].map(cells => cells.map(csvCell).join(',')).join('\r\n') + '\r\n';
  }

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
    const rows = this.visible;
    const best = rows.filter(r => r.final_val != null).reduce((b, r) => (!b || r.final_val < b.final_val ? r : b), null);
    const head = `<thead><tr>
      <th>#</th><th>${t('Time')}</th>
      ${SETTING_COLUMNS.map(c => `<th>${c.label()}</th>`).join('')}
      <th class="num">${t('Train loss')}</th><th class="num">${t('Validation loss')}</th><th class="num">${t('Seconds')}</th><th></th>
    </tr></thead>`;
    const L = { view: esc(t('Show this run')), load: esc(t('Load these settings')), del: esc(t('Delete')), best: esc(t('Lowest validation loss')) };
    const body = rows.map(r => {
      const prev = this.previous(r);
      const cells = SETTING_COLUMNS.map(c => {
        const now = c.get(r.config), before = prev ? c.get(prev.config) : now;
        if (now === before) return `<td>${esc(now)}</td>`;
        const title = esc(t('Before: {v}', { v: before }));
        return `<td class="changed" title="${title}">${esc(now)}</td>`;
      }).join('');
      const status = STATUS()[r.status];
      const time = new Date(r.created_at);
      const short = localDate(r.created_at) === localDate(new Date())
        ? { hour: '2-digit', minute: '2-digit' } : { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' };
      return `<tr data-id="${r.id}" class="${r.id === this.activeId ? 'active' : ''}">
        <td class="seq">${r.seq}</td>
        <td class="time" title="${time.toLocaleString(LANG)}">${time.toLocaleString(LANG, short)}</td>
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
    const none = this.rows.length ? t('No runs in this period.') : t('No runs yet.');
    const empty = `<tbody><tr><td colspan="10" class="empty">${none}</td></tr></tbody>`;
    this.table.innerHTML = head + (rows.length ? `<tbody>${body}</tbody>` : empty);
    this.onRender?.(rows.length, this.rows.length);
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

// 'YYYY-MM-DD' / 'YYYY-MM-DD HH:MM:SS' in the viewer's time zone (dates in the filter are local too)
export function localDate(when) {
  const d = new Date(when), p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

function localDateTime(when) {
  const d = new Date(when), p = n => String(n).padStart(2, '0');
  return `${localDate(d)} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function csvCell(v) {
  if (v == null) return '';
  if (typeof v === 'number') return String(v);
  let s = String(v);
  if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`;       // a custom formula like "=..." must not run in Excel
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
