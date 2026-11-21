// node --test tests/js   (the 2-input plot's wheel and pinch zoom, and that a pinch does not rotate: player.js)
import { test } from 'node:test';
import assert from 'node:assert/strict';

globalThis.matchMedia = () => ({ matches: false, addEventListener() {} });
globalThis.window = { I18N: {} };
globalThis.document = { documentElement: { lang: 'ko' } };
const { Player } = await import('../../app/static/js/player.js');
const { DEFAULT_VIEW, ZOOM_RANGE } = await import('../../app/static/js/plot.js');

function plot() {
  const canvas = new EventTarget();
  canvas.setPointerCapture = () => {};
  const view = { ...DEFAULT_VIEW }, self = { render() {} };
  Player.prototype.enableRotate.call(self, canvas, view, () => true);
  Player.prototype.enableZoom.call(self, canvas, view, () => true);
  const fire = (type, props) => {
    const e = Object.assign(new Event(type, { cancelable: true }), props);
    canvas.dispatchEvent(e);
    return e;
  };
  const finger = (type, id, x, y) => fire(type, { pointerId: id, pointerType: 'touch', clientX: x, clientY: y });
  return { view, fire, finger };
}

test('the wheel zooms within the range and keeps the page from scrolling', () => {
  const { view, fire } = plot();
  const e = fire('wheel', { deltaY: -100, deltaMode: 0 });
  assert.ok(e.defaultPrevented && view.zoom > 1);
  for (let i = 0; i < 50; i++) fire('wheel', { deltaY: -500, deltaMode: 0 });
  assert.equal(view.zoom, ZOOM_RANGE[1]);
  for (let i = 0; i < 50; i++) fire('wheel', { deltaY: 3, deltaMode: 1 });      // lines (Firefox)
  assert.equal(view.zoom, ZOOM_RANGE[0]);
  fire('dblclick', {});
  assert.equal(view.zoom, 1);
});

test('two fingers pinch to zoom without turning the plot', () => {
  const { view, finger } = plot();
  const { az, el } = view;
  finger('pointerdown', 1, 100, 100);
  finger('pointerdown', 2, 200, 100);
  finger('pointermove', 2, 250, 100);              // spread 100 → 150
  assert.ok(Math.abs(view.zoom - 1.5) < 1e-9);
  finger('pointermove', 1, 50, 140);
  assert.deepEqual([view.az, view.el], [az, el]);
  const pinched = view.zoom;
  finger('pointerup', 2, 250, 100);
  finger('pointermove', 1, 10, 140);               // one finger left: neither zooms nor turns until lifted
  assert.ok(view.zoom === pinched && view.az === az);
  finger('pointerup', 1, 10, 140);
  finger('pointerdown', 3, 100, 100);              // a new one-finger drag turns it again
  finger('pointermove', 3, 150, 100);
  assert.ok(view.az !== az);
});

test('the square button turns the plot back to the standard view, and pressing it does not turn the plot', () => {
  const canvas = new EventTarget();
  Object.assign(canvas, { setPointerCapture() {}, style: {}, getBoundingClientRect: () => ({ left: 0, top: 0 }) });
  const view = { az: 1.2, el: 1.0, zoom: 2 };
  const self = { canvas, view, render() {}, predLayout: { resetView: { x: 300, y: 200, w: 30, h: 30 } } };
  Player.prototype.enableResetView.call(self);
  Player.prototype.enableRotate.call(self, canvas, view, () => true);
  const fire = (type, x, y) => canvas.dispatchEvent(Object.assign(new Event(type), { pointerId: 1, pointerType: 'mouse', clientX: x, clientY: y }));
  fire('pointermove', 310, 210);
  assert.equal(canvas.style.cursor, 'pointer');
  fire('pointerdown', 310, 210);
  fire('pointermove', 400, 260);                   // the drag that follows does not turn it
  assert.deepEqual(view, { ...DEFAULT_VIEW, zoom: 1 });
  fire('pointerup', 400, 260);
  fire('pointerdown', 100, 100);                   // elsewhere a drag turns it as before
  fire('pointermove', 150, 100);
  assert.notEqual(view.az, DEFAULT_VIEW.az);
});
