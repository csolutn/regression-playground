// History table: newest run on top; a cell is highlighted when it differs from the run before it.
// Filter icons next to "Time" (a date range) and "① Data" (one of the data sets used so far) limit the
// rows shown; the arrow next to "Validation loss" sorts by it (lowest first → highest first → newest first);
// ★ marks the lowest validation loss of each data set; the shown rows can be saved as CSV.
// The server sends the newest runs a page at a time: `older` is the run just below the loaded ones
// (null once everything is loaded), `total` the count of all runs.
import { t } from './i18n.js';
import { fmtLoss } from './plot.js';
import { dataKey, dataLabel, dataSummary, lossSummary, modelSummary, optimSummary } from './describe.js';

const FUNNEL = '<svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true"><path d="M2 3h12l-4.5 5.5V13l-3 1.5V8.5z" fill="currentColor"/></svg>';

// The four setting columns follow the pipeline ① → ④; `get` turns a run's settings into text.
export const SETTING_COLUMNS = [
  { key: 'data', label: () => t('① Data'), get: dataSummary },
  { key: 'model', label: () => t('② Model'), get: modelSummary },
  { key: 'loss', label: () => t('③ Loss'), get: lossSummary },
  { key: 'optim', label: () => t('④ Optimization'), get: optimSummary },
];

const NEXT_SORT = { '': 'asc', asc: 'desc', desc: '' };
const SORT_ARROW = { '': '↕', asc: '↑', desc: '↓' };

const LANG = document.documentElement.lang || undefined;
const STATUS = () => ({ timeout: t('time limit'), diverged: t('diverged') });

export class History {
  constructor(table, { readonly = false, popup, onView, onLoad, onDelete, onNeedAll }) {
    this.table = table;
    this.readonly = readonly;
    this.popup = popup;
    this.handlers = { onView, onLoad, onDelete, onNeedAll };    // onNeedAll: loads every run (before a filter)
    this.rows = [];
    this.older = null;
    this.total = 0;
    this.activeId = null;
    this.filter = { from: '', to: '', data: '' };      // local dates 'YYYY-MM-DD' and a dataKey; '' = no limit
    this.sort = '';                                     // by validation loss: '' (newest first) | 'asc' | 'desc'
    this.onRender = null;
    table.addEventListener('click', e => this.onClick(e));
    this.setupPopup();
  }

  setRows(rows, { older = null, total = rows.length } = {}) { this.rows = rows; this.older = older; this.total = total; this.render(); }
  addOlder(rows, older, total) { this.rows.push(...rows); this.older = older; this.total = total; this.render(); }
  add(row) { this.rows.unshift(row); this.total++; this.render(); }
  remove(id) {
    const n = this.rows.length;
    this.rows = this.rows.filter(r => r.id !== id);
    this.total -= n - this.rows.length;
    this.render();
  }
  setActive(id) { this.activeId = id; this.render(); }
  setFilter(filter) { this.filter = { ...this.filter, ...filter }; this.render(); }
  setSort(sort) { this.sort = sort; this.render(); }
  // "before" is always the run just before, even when the filter hides it
  previous(row) {
    const i = this.rows.indexOf(row);
    return i < 0 ? null : this.rows[i + 1] || this.older;
  }

  get filtered() { return !!(this.filter.from || this.filter.to || this.filter.data); }
  get visible() {
    const { from, to, data } = this.filter;
    const rows = this.rows.filter(r => {
      const d = localDate(r.created_at);
      return (!from || d >= from) && (!to || d <= to) && (!data || dataKey(r.config) === data);
    });
    if (!this.sort) return rows;
    const sign = this.sort === 'asc' ? 1 : -1;       // runs without a validation loss go last; ties stay newest first
    return rows.sort((a, b) => (a.final_val == null) - (b.final_val == null) || sign * (a.final_val - b.final_val));
  }

  // data (formulas / CSV files) used so far, most recent first: dataKey → { label, count }
  dataSets() {
    const sets = new Map();
    for (const r of this.rows) {
      const key = dataKey(r.config);
      if (sets.has(key)) sets.get(key).count++;
      else sets.set(key, { label: dataLabel(r.config), count: 1 });
    }
    return sets;
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
    if (e.target.closest('button[data-sort]')) {
      Promise.resolve(this.handlers.onNeedAll?.()).then(() => this.setSort(NEXT_SORT[this.sort]));
      return;
    }
    const icon = e.target.closest('button[data-filter]');
    if (icon) {
      const kind = icon.dataset.filter;       // loading the rest redraws the table, so find the icon again
      Promise.resolve(this.handlers.onNeedAll?.())
        .then(() => this.openPopup(kind, this.table.querySelector(`button[data-filter="${kind}"]`)));
      return;
    }
    const btn = e.target.closest('button[data-act]');
    const tr = e.target.closest('tr[data-id]');
    if (!tr) return;
    const row = this.rows.find(r => r.id === +tr.dataset.id);
    if (!btn) { this.handlers.onView?.(row); return; }
    const fn = { view: 'onView', load: 'onLoad', del: 'onDelete' }[btn.dataset.act];
    this.handlers[fn]?.(row);
  }

  render() {
    if (this.filter.data && !this.dataSets().has(this.filter.data)) this.filter.data = '';   // its runs were deleted
    const rows = this.visible;
    const bestOf = new Map();                   // dataKey → the shown run with the lowest validation loss
    for (const r of rows) {
      if (r.final_val == null) continue;
      const key = dataKey(r.config), b = bestOf.get(key);
      if (!b || r.final_val < b.final_val) bestOf.set(key, r);
    }
    const best = new Set(bestOf.values());
    const icon = (kind, on, label) =>
      `<button type="button" class="th-filter${on ? ' on' : ''}" data-filter="${kind}" title="${esc(label)}" aria-label="${esc(label)}" aria-haspopup="dialog">${FUNNEL}</button>`;
    const sortLabel = esc(t('Sort by validation loss'));
    const sortButton = `<button type="button" class="th-filter th-sort${this.sort ? ' on' : ''}" data-sort title="${sortLabel}" aria-label="${sortLabel}">${SORT_ARROW[this.sort]}</button>`;
    const head = `<thead><tr>
      <th>#</th><th>${t('Time')} ${icon('time', this.filter.from || this.filter.to, t('Filter by date'))}</th>
      ${SETTING_COLUMNS.map(c => `<th>${c.label()}${c.key === 'data' ? ' ' + icon('data', this.filter.data, t('Filter by data')) : ''}</th>`).join('')}
      <th class="num">${t('Train loss')}</th>
      <th class="num" aria-sort="${{ '': 'none', asc: 'ascending', desc: 'descending' }[this.sort]}">${t('Validation loss')} ${sortButton}</th>
      <th class="num">${t('Seconds')}</th><th></th>
    </tr></thead>`;
    const L = { view: esc(t('Show this run')), load: esc(t('Load these settings')), del: esc(t('Delete')), best: esc(t('Lowest validation loss for this data')) };
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
        <td class="num ${best.has(r) ? 'best' : ''}">${fmtLoss(r.final_val)}${best.has(r) ? ` <span class="star" title="${L.best}">★</span>` : ''}${status ? ` <span class="pill pill-warn">${status}</span>` : ''}</td>
        <td class="num">${r.duration.toFixed(1)}</td>
        <td class="actions">
          <button type="button" class="btn btn-icon btn-ghost" data-act="view" title="${L.view}">▶</button>
          ${this.readonly ? '' : `<button type="button" class="btn btn-icon btn-ghost" data-act="load" title="${L.load}">⟲</button>
          <button type="button" class="btn btn-icon btn-ghost" data-act="del" title="${L.del}">🗑</button>`}
        </td></tr>`;
    }).join('');
    const none = this.rows.length ? t('No runs match the filter.') : t('No runs yet.');
    const empty = `<tbody><tr><td colspan="10" class="empty">${none}</td></tr></tbody>`;
    this.table.innerHTML = head + (rows.length ? `<tbody>${body}</tbody>` : empty);
    this.onRender?.(rows.length, this.rows.length);
  }

  // ---------- filter popup under the icon; a click outside or Esc closes it ----------

  setupPopup() {
    const pop = this.popup;
    if (!pop) return;
    document.addEventListener('pointerdown', e => {
      if (!pop.hidden && !pop.contains(e.target) && !e.target.closest('button[data-filter]')) this.closePopup();
    });
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && !pop.hidden) this.closePopup(); });
    pop.addEventListener('change', e => {
      const f = e.target.dataset.f;
      if (!f) return;
      this.setFilter({ [f]: e.target.value });
      if (f === 'data') { this.closePopup(); return; }
      pop.querySelector('[data-f=to]').min = this.filter.from;      // keep from ≤ to
      pop.querySelector('[data-f=from]').max = this.filter.to;
    });
    pop.addEventListener('click', e => {
      const act = e.target.closest('button[data-pop]')?.dataset.pop;
      if (act === 'today') { const d = localDate(new Date()); this.setFilter({ from: d, to: d }); this.closePopup(); }
      if (act === 'all-dates') { this.setFilter({ from: '', to: '' }); this.closePopup(); }
    });
  }

  openPopup(kind, anchor) {
    const pop = this.popup;
    if (!pop) return;
    if (!pop.hidden && pop.dataset.kind === kind) { this.closePopup(); return; }
    pop.dataset.kind = kind;
    if (kind === 'time') this.fillTimePopup();
    else {
      const sets = this.dataSets();
      const title = esc(t('Filter by data')), all = esc(t('All data'));
      const opts = [...sets].map(([key, s]) =>
        `<option value="${esc(key)}" ${key === this.filter.data ? 'selected' : ''}>${esc(s.label)} (${s.count})</option>`);
      pop.innerHTML = `<label class="filter-pop-title">${title}</label>
        <select data-f="data"><option value="">${all} (${this.rows.length})</option>${opts.join('')}</select>`;
    }
    pop.hidden = false;
    // below the icon, inside the card (the popup's offset parent)
    const box = pop.offsetParent.getBoundingClientRect(), a = anchor.getBoundingClientRect();
    pop.style.top = `${a.bottom - box.top + 6}px`;
    pop.style.left = `${Math.max(8, Math.min(a.left - box.left - 8, box.width - pop.offsetWidth - 8))}px`;
    pop.querySelector('input, select')?.focus();
  }

  fillTimePopup() {
    const { from, to } = this.filter;
    const L = { title: esc(t('Filter by date')), from: esc(t('From date')), to: esc(t('To date')), today: esc(t('Today')), all: esc(t('All')) };
    this.popup.innerHTML = `<label class="filter-pop-title">${L.title}</label>
      <div class="filter-pop-row">
        <input type="date" data-f="from" value="${from}" max="${to}" aria-label="${L.from}">
        <span class="muted">–</span>
        <input type="date" data-f="to" value="${to}" min="${from}" aria-label="${L.to}">
      </div>
      <div class="filter-pop-row">
        <button type="button" class="btn btn-ghost btn-sm" data-pop="today">${L.today}</button>
        <button type="button" class="btn btn-ghost btn-sm" data-pop="all-dates">${L.all}</button>
      </div>`;
  }

  closePopup() { this.popup.hidden = true; }
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
