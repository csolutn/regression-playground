// MP4 export, made entirely in the browser (no server work).
// Layout follows a vertical short video (1080×1920): title, prediction plot, loss curve.
// WebCodecs + mp4-muxer when available; otherwise MediaRecorder (may produce WebM).
import { t } from './i18n.js';
import { dataSummary } from './describe.js';
import { frameMs } from './player.js';
import { DEFAULT_VIEW, FONT, LIGHT, drawLoss, drawPrediction } from './plot.js';

const W = 1080, H = 1920, S = 2;          // S: plot scale (fonts and lines) in the video
const FIRST_HOLD_US = 500_000, LAST_HOLD_US = 2_000_000;
const MUXER_SRC = '/static/vendor/mp4-muxer.min.js';

export async function exportVideo(run, { title, speed = 1, log = false, view = DEFAULT_VIEW, onProgress = () => {} }) {
  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const n = run.frames.length;
  const hold = Math.round(frameMs(n, speed) * 1000);
  const durations = run.frames.map((_, i) => hold + (i === 0 ? FIRST_HOLD_US : 0) + (i === n - 1 ? LAST_HOLD_US : 0));
  const draw = i => drawVideoFrame(ctx, run, i, title, log, view);

  if ('VideoEncoder' in window) {
    const blob = await encodeWebCodecs(canvas, draw, durations, onProgress);
    if (blob) return { blob, ext: 'mp4' };
  }
  return encodeMediaRecorder(canvas, draw, durations, onProgress);
}

// One video frame on its own canvas (used for checks and previews)
export function videoFrameCanvas(run, i, { title, log = false, view = DEFAULT_VIEW } = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  drawVideoFrame(canvas.getContext('2d'), run, i, title, log, view);
  return canvas;
}

export function saveBlob(blob, filename) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}

function drawVideoFrame(ctx, run, i, title, log, view) {
  const th = LIGHT;
  ctx.fillStyle = th.bg;
  ctx.fillRect(0, 0, W, H);

  // title (wrapped, up to 3 lines) and a data line under it
  ctx.fillStyle = th.text;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  let size = 64, lines;
  do { ctx.font = `800 ${size}px ${FONT}`; lines = wrap(ctx, title, W - 120); size -= 4; } while (lines.length > 3 && size > 36);
  const lh = (size + 4) * 1.2, top = 250 - (lines.length * lh) / 2 + lh * 0.8;
  lines.forEach((l, k) => ctx.fillText(l, W / 2, top + k * lh));
  ctx.fillStyle = th.muted;
  ctx.font = `400 30px ${FONT}`;
  ctx.fillText(dataSummary(run.config), W / 2, 370);

  drawPrediction(ctx, { x: 40, y: 420, w: W - 80, h: 980 }, run, i, th, S, { view });
  drawLoss(ctx, { x: 40, y: 1450, w: W - 80, h: 340 }, run, run.frames[i].step, th, S, { log });

  ctx.fillStyle = th.muted;
  ctx.font = `400 24px ${FONT}`;
  ctx.textAlign = 'center';
  ctx.fillText(window.APP_NAME || '', W / 2, H - 50);
}

function wrap(ctx, text, maxWidth) {
  const words = String(text).split(/\s+/);
  const lines = [];
  let line = '';
  for (const w of words) {
    const next = line ? `${line} ${w}` : w;
    if (ctx.measureText(next).width > maxWidth && line) { lines.push(line); line = w; } else line = next;
  }
  if (line) lines.push(line);
  return lines;
}

let muxerLoading = null;
function loadMuxer() {
  if (window.Mp4Muxer) return Promise.resolve();
  muxerLoading ||= new Promise((resolve, reject) => {
    const el = document.createElement('script');
    el.src = MUXER_SRC;
    el.onload = resolve;
    el.onerror = () => reject(new Error('mp4-muxer failed to load'));
    document.head.append(el);
  });
  return muxerLoading;
}

async function encodeWebCodecs(canvas, draw, durations, onProgress) {
  let config = null;
  for (const codec of ['avc1.640028', 'avc1.4d0028', 'avc1.42e028']) {   // H.264 High / Main / Baseline, level 4.0
    const c = { codec, width: W, height: H, bitrate: 4_000_000, framerate: 30 };
    try { if ((await VideoEncoder.isConfigSupported(c)).supported) { config = c; break; } } catch { /* try next */ }
  }
  if (!config) return null;
  await loadMuxer();
  const { Muxer, ArrayBufferTarget } = window.Mp4Muxer;
  const muxer = new Muxer({ target: new ArrayBufferTarget(), video: { codec: 'avc', width: W, height: H }, fastStart: 'in-memory' });
  let failure = null;
  const encoder = new VideoEncoder({ output: (chunk, meta) => muxer.addVideoChunk(chunk, meta), error: e => { failure = e; } });
  encoder.configure(config);

  let timestamp = 0;
  for (let i = 0; i < durations.length; i++) {
    if (failure) throw failure;
    draw(i);
    const frame = new VideoFrame(canvas, { timestamp, duration: durations[i] });
    encoder.encode(frame, { keyFrame: i % 20 === 0 });
    frame.close();
    timestamp += durations[i];
    onProgress((i + 1) / durations.length);
    while (encoder.encodeQueueSize > 4) await new Promise(r => setTimeout(r, 5));
    if (i % 5 === 0) await new Promise(r => setTimeout(r, 0));   // keep the page responsive
  }
  await encoder.flush();
  if (failure) throw failure;
  encoder.close();
  muxer.finalize();
  return new Blob([muxer.target.buffer], { type: 'video/mp4' });
}

// Real-time fallback: plays the frames while recording the canvas
async function encodeMediaRecorder(canvas, draw, durations, onProgress) {
  const type = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm']
    .find(m => window.MediaRecorder?.isTypeSupported(m));
  if (!type) throw new Error(t('This browser cannot record video. Try the latest Chrome, Edge or Safari.'));
  draw(0);
  const stream = canvas.captureStream(30);
  const rec = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 4_000_000 });
  const chunks = [];
  rec.ondataavailable = e => e.data.size && chunks.push(e.data);
  const stopped = new Promise(r => { rec.onstop = r; });
  rec.start();
  for (let i = 0; i < durations.length; i++) {
    draw(i);
    onProgress((i + 1) / durations.length);
    await new Promise(r => setTimeout(r, durations[i] / 1000));
  }
  rec.stop();
  await stopped;
  stream.getTracks().forEach(tr => tr.stop());
  const mime = type.split(';')[0];
  return { blob: new Blob(chunks, { type: mime }), ext: mime === 'video/mp4' ? 'mp4' : 'webm' };
}
