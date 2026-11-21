// A browser run trained again in this browser (in a Web Worker), for its picture on the teacher's student page:
// only its results are stored on the server. The same settings and seed on the same data make the same numbers.
import { replayRun } from './train.js';
import { drawable, resultOf } from './report.js';
import { replayNote } from './replay-here.js';

export async function replayResult(entry, metas, onProgress, signal) {
  const run = drawable(entry, metas);
  const made = await replayRun(run.config, run.meta, entry.row.steps, onProgress, signal);
  return resultOf(made, replayNote(entry, made.end));
}
