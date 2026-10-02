// Trains one run off the page's thread, so the page keeps animating: posts each event of nn.js's train().
import { train } from './nn.js';

self.onmessage = ({ data: { cfg, meta, timeLimitS } }) => {
  try {
    for (const ev of train(cfg, meta, { timeLimitS })) self.postMessage(ev);
  } catch (err) {
    self.postMessage({ type: 'crash', error: String(err?.message ?? err) });
  }
};
