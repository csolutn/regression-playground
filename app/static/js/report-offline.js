// The HTML backup, opened from the disk: the students on the left, the picked student's results on the right.
// Everything it needs is in the file (window.REPORT; static/js/backup.js wrote it): a browser run is trained
// again here when it is picked (replay-here.js), a server run has its last frame.
import { t } from './i18n.js';
import { Report, idLabel } from './report.js';
import { esc } from './history.js';
import { replayHere } from './replay-here.js';

const data = window.REPORT;
const list = document.getElementById('students');
const report = new Report(document.getElementById('report'), {
  metas: data.metas, resolve: (entry, onProgress, signal) => replayHere(entry, data.metas, onProgress, signal),
});

list.innerHTML = `<thead><tr><th>${esc(t('Student ID'))}</th><th>${esc(t('Name'))}</th><th class="num">${esc(t('Runs'))}</th></tr></thead>
  <tbody>${data.students.map((s, i) => `<tr data-i="${i}"><td>${idLabel(s)}</td><td>${esc(s.name)}</td><td class="num">${s.runs.length}</td></tr>`).join('')}</tbody>`;

function pick(i) {
  for (const tr of list.querySelectorAll('tr[data-i]')) tr.classList.toggle('active', +tr.dataset.i === i);
  report.setStudent(data.students[i]);
}
list.addEventListener('click', e => {
  const tr = e.target.closest('tr[data-i]');
  if (tr) pick(+tr.dataset.i);
});
if (data.students.length) pick(0);
