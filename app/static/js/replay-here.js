// A browser run trained again on this page's own thread, a few milliseconds at a time so the page stays usable:
// for the HTML backup, which is one file opened from the disk (it can not start a Web Worker from another file).
// The same settings and seed on the same data make the same numbers as when the student trained it.
import { t } from './i18n.js';
import { TRAINER_VERSION, train } from './nn.js';
import { drawable, resultOf } from './report.js';

const REPLAY_TIME_LIMIT_S = 600;      // a replay may take longer than the run did (another computer)
const SLICE_MS = 25;                  // training this long, then the page gets a turn

// why a replay may not look like the run did: another trainer, other data, or numbers a little off
export function replayNote(entry, end) {
  const saved = entry.row.final_val, now = end.final_val;
  const differs = saved != null && now != null && Math.abs(now - saved) > 1e-3 * Math.abs(saved);
  return entry.replay.trainer !== TRAINER_VERSION || entry.replay.data_changed
    ? t('The training code has changed since this run: its replay may differ from what it showed then.')
    : differs ? t('Trained again in this browser: the numbers differ a little from the saved ones.') : null;
}

export async function replayHere(entry, metas, onProgress, signal) {
  const run = drawable(entry, metas);
  const cfg = { ...run.config, epochs: Math.max(1, entry.row.steps) };     // up to the epoch it ended at
  const made = { frames: [], loss: { steps: [], train: [], val: [] }, end: null };
  let sliceEnd = performance.now() + SLICE_MS;
  for (const ev of train(cfg, run.meta, { timeLimitS: REPLAY_TIME_LIMIT_S })) {
    if (ev.type === 'frame') {
      made.frames = [ev];                                    // only the last one is drawn
      onProgress(ev.step / cfg.epochs);
    } else if (ev.type === 'loss') {
      for (const k of ['steps', 'train', 'val']) made.loss[k].push(...ev[k]);
    } else if (ev.type === 'end') {
      made.end = ev;
    }
    if (performance.now() > sliceEnd) {
      await new Promise(resolve => setTimeout(resolve));
      if (signal.aborted) throw new DOMException('Training stopped', 'AbortError');
      sliceEnd = performance.now() + SLICE_MS;
    }
  }
  return resultOf(made, replayNote(entry, made.end));
}
