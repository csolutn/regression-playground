// Where a run trains. Linear and neural network runs train in this browser (train-worker.js), which
// spares the server, unless the server answers that it is much faster there (options.trains_in_browser);
// tree models train on the server. Either way onEvent gets the same events, ending with 'saved'.
// A guest trains only in the browser, and the run is not saved: its history row is made here.
import { t } from './i18n.js';
import { api } from './api.js';
import { isTree } from './describe.js';

let guestSeq = 0;

export async function trainRun(cfg, onEvent, signal, { guest = false } = {}) {
  if (guest && isTree(cfg)) throw new Error(t('Tree models train on the server: log in to use them.'));
  if (!isTree(cfg)) {
    const prep = await api.prepare(cfg, signal);
    if (prep.browser) return trainHere(cfg, prep, onEvent, signal, guest);
    if (guest) throw new Error(t('This run is too big for the browser. Make the network or the epochs smaller, or log in to train it on the server.'));
  }
  return api.train(cfg, onEvent, signal);
}

async function trainHere(cfg, { meta, time_limit_s: timeLimitS }, onEvent, signal, guest) {
  const stored = { frames: [], loss: { steps: [], train: [], val: [] }, end: null };
  await new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./train-worker.js', import.meta.url), { type: 'module' });
    const finish = (settle, value) => {
      worker.terminate();
      signal.removeEventListener('abort', stop);
      settle(value);
    };
    const stop = () => finish(reject, new DOMException('Training stopped', 'AbortError'));
    if (signal.aborted) return stop();
    signal.addEventListener('abort', stop);
    worker.onmessage = ({ data: ev }) => {
      if (ev.type === 'crash') return finish(reject, new Error(ev.error));
      if (ev.type === 'frame') stored.frames.push({ step: ev.step, pred: ev.pred, train: ev.train, val: ev.val });
      else if (ev.type === 'loss') for (const k of ['steps', 'train', 'val']) stored.loss[k].push(...ev[k]);
      else if (ev.type === 'end') stored.end = ev;
      onEvent(ev);
      if (ev.type === 'end') finish(resolve);
    };
    worker.onerror = e => { e.preventDefault(); finish(reject, new Error(e.message || 'Training failed.')); };
    worker.postMessage({ cfg, meta, timeLimitS });
  });
  const { run } = guest ? { run: guestRow(cfg, stored.end) }
    : await api.saveRun({ config: cfg, ...stored });                   // not abortable: the run is finished
  onEvent({ type: 'saved', run });
}

// the row the server would have returned (Run.to_row), for a guest's run that is not saved
function guestRow({ csv_text, ...config }, end) {     // like the server's rows: without the CSV text
  guestSeq++;
  return { id: -guestSeq, seq: guestSeq, created_at: new Date().toISOString(), model: config.model, status: end.status,
           final_train: end.final_train, final_val: end.final_val, steps: end.steps, duration: end.duration, config };
}
