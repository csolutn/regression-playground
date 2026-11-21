// The teacher's page of one student: the results of their runs (report.js), each browser run trained again
// here when it is picked (nothing of it is stored).
import { Report } from './report.js';
import { replayResult } from './report-replay.js';
import { toast } from './toast.js';

const root = document.getElementById('report');
try {
  const res = await fetch(`/teacher/report.json?student=${root.dataset.student}`);
  if (!res.ok) throw new Error(`Server error (${res.status})`);
  const data = await res.json();
  const report = new Report(root, { metas: data.metas, resolve: (entry, onProgress, signal) => replayResult(entry, data.metas, onProgress, signal) });
  report.setStudent(data.students[0]);
} catch (err) {
  toast(err.message, 'error');
}
