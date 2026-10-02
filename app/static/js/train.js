// Where a run trains. Linear and neural network runs train in this browser (train-worker.js), which
// spares the server, unless the server answers that it is much faster there (options.trains_in_browser);
// tree models train on the server. Either way onEvent gets the same events, ending with 'saved'.
import { api } from './api.js';
import { isTree } from './describe.js';

export async function trainRun(cfg, onEvent, signal) {
  if (!isTree(cfg)) {
    const prep = await api.prepare(cfg, signal);
    if (prep.browser) return trainHere(cfg, prep, onEvent, signal);
  }
  return api.train(cfg, onEvent, signal);
}

async function trainHere(cfg, { meta, time_limit_s: timeLimitS }, onEvent, signal) {
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
  const { run } = await api.saveRun({ config: cfg, ...stored });    // not abortable: the run is finished
  onEvent({ type: 'saved', run });
}
