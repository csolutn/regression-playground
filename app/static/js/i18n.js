// Translations come from the server catalog (app/translations). Source strings are English.
const catalog = window.I18N || {};

export function t(msg, params) {
  let s = catalog[msg] || msg;
  if (params) s = s.replace(/\{(\w+)\}/g, (m, k) => (k in params ? params[k] : m));
  return s;
}
