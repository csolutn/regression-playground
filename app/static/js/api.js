// Server calls. POST /api/train streams one JSON event per line (see app/ml/trainer.py).
import { t } from './i18n.js';

export class ApiError extends Error {
  constructor(messages) {
    super(messages.join('\n'));
    this.messages = messages;
  }
}

async function check(res) {
  if (res.status === 401) { location.href = '/login?next=' + encodeURIComponent(location.pathname); throw new ApiError([t('Please log in again.')]); }
  if (res.ok) return res;
  let messages;
  try { messages = (await res.json()).errors; } catch { /* not JSON */ }
  throw new ApiError(messages?.length ? messages : [t('Server error ({status})', { status: res.status })]);
}

async function getJSON(url, opts = {}) {
  const res = await check(await fetch(url, opts));
  return res.status === 204 ? null : res.json();
}

const post = body => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

export const api = {
  runs: userId => getJSON('/api/runs' + (userId ? `?user=${userId}` : '')),
  run: id => getJSON(`/api/runs/${id}`),
  deleteRun: id => getJSON(`/api/runs/${id}`, { method: 'DELETE' }),
  preview: cfg => getJSON('/api/preview', post(cfg)),

  async train(cfg, onEvent, signal) {
    const res = await check(await fetch('/api/train', { ...post(cfg), signal }));
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i);
        buf = buf.slice(i + 1);
        if (line.trim()) onEvent(JSON.parse(line));
      }
    }
  },
};
