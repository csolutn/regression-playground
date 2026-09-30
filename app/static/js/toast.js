let timer = null;

export function toast(message, kind = 'info') {
  const el = document.getElementById('toast');
  if (!el) { alert(message); return; }
  el.textContent = message;
  el.className = `toast toast-${kind}`;
  el.hidden = false;
  clearTimeout(timer);
  timer = setTimeout(() => { el.hidden = true; }, kind === 'error' ? 7000 : 3500);
}
