// The teacher page's dialogs: add a teacher, the history backup, and "delete all students".
//
// The backup is one HTML file with the students' runs, drawn as on the student page (report.js), that works
// offline. Nothing is trained here: the file has the data and the trainer (nn.js), and trains a browser run again
// when it is picked there (replay-here.js). It is put together from /teacher/report.json, the page's CSS and
// report-offline.js with what it imports, joined into one script (/teacher/report-bundle.js). Safari does not save
// a file the page starts by itself after waiting for the server, so the teacher clicks "Save the file".
import { t } from './i18n.js';
import { toast } from './toast.js';

// ---------- dialogs: [data-open="<id>"] opens one, [data-close] in it closes it ----------

for (const btn of document.querySelectorAll('[data-open]')) {
  btn.addEventListener('click', () => document.getElementById(btn.dataset.open).showModal());
}
for (const btn of document.querySelectorAll('dialog [data-close]')) {
  btn.addEventListener('click', () => btn.closest('dialog').close());
}

// ---------- the backup file ----------

async function getText(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(t('Server error ({status})', { status: res.status }));
  return res.text();
}

// report.json, telling `onProgress(got, total)` (bytes) as it comes: its size is in X-Report-Size, since
// Content-Length is dropped when the response is compressed on the way (and counts the compressed bytes)
async function getReport(url, signal, onProgress) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(t('Server error ({status})', { status: res.status }));
  const total = Number(res.headers.get('X-Report-Size')) || 0;
  if (!total || !res.body) return res.json();
  const reader = res.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(Math.min(got, total), total);
  }
  return JSON.parse(await new Blob(chunks).text());
}

// { url, name, size } of the backup of the students report.json gives for `query`; `bar` shows how far it got
async function makeBackup(query, signal, bar, text) {
  bar.wait();
  text.textContent = t('Gathering the runs on the server…');
  const [data, css, script] = await Promise.all([
    getReport(`/teacher/report.json${query}`, signal, (got, total) => {
      bar.set(got / total);
      text.textContent = t('Getting the data… {got} / {total} MB', { got: mb(got), total: mb(total) });
    }),
    getText(document.querySelector('link[rel=stylesheet]').href, signal),
    getText('/teacher/report-bundle.js', signal),
  ]);
  text.textContent = t('Making the file…');
  const date = new Date().toLocaleDateString('sv');                  // YYYY-MM-DD
  const who = data.teacher.prefix || data.teacher.name;
  const blob = new Blob([page(data, css, script, date)], { type: 'text/html;charset=utf-8' });
  bar.done();
  return { url: URL.createObjectURL(blob), name: `yhat-backup-${who}-${date}.html`, size: blob.size };
}

const mb = size => (size / 1e6).toFixed(1);

// the progress bar in a dialog: wait() slides while the amount is not known, set(0..1), done() fills it, hide()
function progressBar(dialog) {
  const el = dialog.querySelector('[data-role="progress"]'), fill = el.firstElementChild;
  return {
    wait() { el.hidden = false; el.classList.add('is-waiting'); fill.style.width = ''; },
    set(f) { el.hidden = false; el.classList.remove('is-waiting'); fill.style.width = `${(100 * f).toFixed(1)}%`; },
    done() { this.set(1); },
    hide() { el.hidden = true; el.classList.remove('is-waiting'); fill.style.width = '0'; },
  };
}

// a link that saves the file when the teacher clicks it
function offer(link, file) {
  link.href = file.url;
  link.download = file.name;
  link.hidden = false;
}

function withdraw(link) {
  if (link.href) URL.revokeObjectURL(link.href);
  link.removeAttribute('href');
  link.hidden = true;
}

// "Back up the history": every student I see, or one teacher's (the system admin's row buttons)
{
  const dialog = document.getElementById('backup-dialog');
  const text = dialog.querySelector('[data-role="backup-text"]'), save = dialog.querySelector('[data-role="backup-save"]');
  const bar = progressBar(dialog);
  let running = null;
  for (const btn of document.querySelectorAll('[data-backup]')) btn.addEventListener('click', () => backup(btn.dataset.backup));
  dialog.querySelector('[data-role="backup-cancel"]').addEventListener('click', () => dialog.close());
  save.addEventListener('click', () => setTimeout(() => dialog.close()));          // after the browser took the link
  dialog.addEventListener('close', () => { running?.abort(); withdraw(save); bar.hide(); });

  async function backup(teacherId) {
    if (running) return;
    const ac = running = new AbortController();
    dialog.showModal();
    try {
      const file = await makeBackup(teacherId ? `?teacher=${teacherId}` : '', ac.signal, bar, text);
      offer(save, file);
      text.textContent = t('The file is ready ({size} MB).', { size: mb(file.size) });
    } catch (err) {
      if (err.name !== 'AbortError') { dialog.close(); toast(err.message, 'error'); }
    } finally {
      if (running === ac) running = null;
    }
  }
}

// "Delete all students": type "delete", or "back up and delete" to save their backup first
{
  const dialog = document.getElementById('delete-all-dialog');
  const form = dialog.querySelector('form'), input = form.elements.confirm_text;
  const ok = dialog.querySelector('[data-role="delete-all-ok"]'), save = dialog.querySelector('[data-role="delete-all-save"]');
  const text = dialog.querySelector('[data-role="delete-all-text"]');
  const bar = progressBar(dialog);
  const words = { delete: JSON.parse(input.dataset.delete), backup: JSON.parse(input.dataset.backupWords) };
  const typed = () => input.value.split(/\s+/).filter(Boolean).join(' ').toLowerCase();
  const mode = () => (words.delete.includes(typed()) ? 'delete' : words.backup.includes(typed()) ? 'backup' : null);
  let running = null;
  const okLabel = ok.textContent;

  const reset = () => {
    running?.abort();
    running = null;
    withdraw(save);
    input.value = '';
    input.disabled = false;
    ok.hidden = false;
    ok.disabled = true;
    ok.textContent = okLabel;
    text.textContent = '';
    bar.hide();
  };
  dialog.addEventListener('close', reset);
  input.addEventListener('input', () => {
    ok.disabled = !mode();
    ok.textContent = mode() === 'backup' ? t('Make the backup') : okLabel;
  });
  form.addEventListener('submit', async e => {
    if (mode() === 'delete') return;                                 // the server checks the phrase again
    e.preventDefault();
    if (mode() !== 'backup' || running) return;
    const ac = running = new AbortController();
    input.readOnly = true;
    ok.disabled = true;
    try {
      const file = await makeBackup('?class=1', ac.signal, bar, text);
      offer(save, file);
      ok.hidden = true;
      text.textContent = t('The file is ready ({size} MB). Save it: the students are deleted right after.', { size: mb(file.size) });
    } catch (err) {
      if (err.name !== 'AbortError') { text.textContent = err.message; ok.disabled = false; bar.hide(); }
    } finally {
      input.readOnly = false;
      if (running === ac) running = null;
    }
  });
  save.addEventListener('click', () => {           // the browser saves the file (the link), then the form is sent
    text.textContent = t('Saving the file, then deleting…');
    setTimeout(() => form.submit(), 1500);
  });
}

// ---------- the file ----------

// JSON inside a <script>: no "</script>" or "<!--" can end it early
const json = v => JSON.stringify(v).replace(/</g, '\\u003c');
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

function page(data, css, script, date) {
  const app = window.APP_NAME || 'ŷ Playground';
  const title = `${app} · ${t('Training history')} · ${data.teacher.name} · ${date}`;
  const runs = data.students.reduce((n, s) => n + s.runs.length, 0);
  const heading = `${app} · ${t('Training history')}`;
  const counts = t('{n} students, {runs} runs', { n: data.students.length, runs });    // (outside the template: pybabel)
  const hint = t('Pick a run: this computer trains it again from its settings to draw the result (a few seconds).');
  return `<!doctype html>
<html lang="${esc(document.documentElement.lang)}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)}</title>
<style>${css.replace(/<\/style/gi, '<\\/style')}</style>
</head>
<body class="page-admin page-report">
<main class="admin">
  <header class="admin-head">
    <h1>${esc(heading)}</h1>
    <span class="muted">${esc(data.teacher.name)} · ${esc(counts)} · ${esc(date)}</span>
  </header>
  <p class="note">${esc(hint)}</p>
  <div class="report-layout">
    <section class="card report-students"><div class="table-wrap"><table id="students" class="history admin-table"></table></div></section>
    <div id="report" class="report-main"></div>
  </div>
</main>
<script>window.I18N = ${json(window.I18N || {})}; window.OPTIONS = ${json(window.OPTIONS || {})}; window.REPORT = ${json(data)};</script>
<script>${script.replace(/<\/script/gi, '<\\/script')}</script>
</body>
</html>
`;
}
