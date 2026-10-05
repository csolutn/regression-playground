// Where a run trains. Linear and neural network runs train in this browser (train-worker.js), which
// spares the server, unless the server answers that it is much faster there (options.trains_in_browser);
// tree models train on the server. Either way onEvent gets the same events, ending with 'saved'.
// A browser run is shown as soon as it ends: only its results are saved, meanwhile, and replayRun trains
// it again from its settings whenever it is shown later. A guest's run is not saved: its row is made here.
import { t } from './i18n.js';
import { api } from './api.js';
import { isTree } from './describe.js';
import { TRAINER_VERSION } from './nn.js';

const REPLAY_TIME_LIMIT_S = 600;      // a replay may take longer than the run did (another computer)

let guestSeq = 0;

export async function trainRun(cfg, onEvent, signal, { guest = false } = {}) {
  const notForGuests = hint => new Error(`${t('This feature is not available to guests.')} ${hint}`);
  if (guest && isTree(cfg)) throw notForGuests(t('Log in to train tree models.'));
  if (!isTree(cfg)) {
    const prep = await api.prepare(cfg, signal);
    if (prep.browser) return trainHere(cfg, prep, onEvent, signal, guest);
    if (guest) throw notForGuests(t('With a smaller network or fewer epochs it trains in the browser.'));
  }
  return api.train(cfg, onEvent, signal);
}

async function trainHere(cfg, { meta, time_limit_s: timeLimitS }, onEvent, signal, guest) {
  const { end } = await inWorker(cfg, meta, timeLimitS, onEvent, signal);
  const saving = guest ? Promise.resolve({ run: guestRow(cfg, end) })
    : api.saveRun({ config: cfg, end, trainer: TRAINER_VERSION });      // not abortable: the run is finished
  saving.then(({ run }) => onEvent({ type: 'saved', run }), err => onEvent({ type: 'error', message: err.message }));
}

// A saved browser run trained again from its settings and seed, for its animation, loss curve and loss
// landscape: the same numbers, up to the epoch it ended at (where a time limit stopped it). onProgress(0…1)
export function replayRun(cfg, meta, steps, onProgress, signal) {
  const replay = { ...cfg, epochs: Math.max(1, steps) };
  return inWorker(replay, meta, REPLAY_TIME_LIMIT_S, ev => { if (ev.type === 'frame') onProgress(ev.step / replay.epochs); }, signal);
}

// train() of nn.js in a Web Worker: each event goes to onEvent; resolves with what the run made
function inWorker(cfg, meta, timeLimitS, onEvent, signal) {
  const made = { frames: [], loss: { steps: [], train: [], val: [] }, end: null, params: null };
  return new Promise((resolve, reject) => {
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
      if (ev.type === 'frame') made.frames.push({ step: ev.step, pred: ev.pred, train: ev.train, val: ev.val });
      else if (ev.type === 'loss') for (const k of ['steps', 'train', 'val']) made.loss[k].push(...ev[k]);
      else if (ev.type === 'params') made.params = { init: ev.init, final: ev.final, contrib: ev.contrib, path_epochs: ev.path_epochs, paths: ev.paths };
      else if (ev.type === 'end') made.end = ev;
      onEvent(ev);
      if (ev.type === 'end') finish(resolve, made);
    };
    worker.onerror = e => { e.preventDefault(); finish(reject, new Error(e.message || 'Training failed.')); };
    worker.postMessage({ cfg, meta, timeLimitS });
  });
}

// the row the server would have returned (Run.to_row), for a guest's run that is not saved
function guestRow({ csv_text, ...config }, end) {     // like the server's rows: without the CSV text
  guestSeq++;
  return { id: -guestSeq, seq: guestSeq, created_at: new Date().toISOString(), model: config.model, status: end.status,
           final_train: end.final_train, final_val: end.final_val, steps: end.steps, duration: end.duration, config };
}
