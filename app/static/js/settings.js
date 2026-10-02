// Settings form state. Inputs are bound by their `name` (same keys as app/ml/options.py DEFAULTS).
//   data-show-if="model:neural_net"                 shown when model is neural_net
//   data-show-if="n_inputs:1;function_1d:custom"    all clauses must match (values separated by ,)
//   data-json                                       hidden input holding a JSON value (e.g. hidden_layers)
// Values are kept per user in localStorage (a guest: sessionStorage, this tab only), so a reload keeps
// the experiment (playground.js bumps the key version when the defaults change, so everyone starts from them once).

export class Settings extends EventTarget {
  constructor(form, defaults, storageKey, storage = () => localStorage) {
    super();
    this.form = form;
    this.defaults = defaults;
    this.storageKey = storageKey;
    this.storage = storage;            // a function: reading the storage itself can throw when it is blocked
    this.values = { ...defaults, ...this.load() };
    this.write();
    this.refresh();
    form.addEventListener('input', e => this.onInput(e));
    form.addEventListener('change', e => this.onInput(e));
  }

  get() {
    return structuredClone(this.values);
  }

  set(values) {
    const known = Object.fromEntries(Object.entries(values).filter(([k]) => k in this.defaults));
    Object.assign(this.values, known);
    this.write();
    this.changed(Object.keys(known));
  }

  reset() {
    this.values = structuredClone(this.defaults);
    this.write();
    this.changed(Object.keys(this.defaults));
  }

  onInput(e) {
    const name = e.target.name;
    if (!name || !(name in this.defaults)) return;
    this.values[name] = this.read(name);
    this.showValue(name);
    this.changed([name]);
  }

  changed(keys) {
    this.refresh();
    this.save();
    this.dispatchEvent(new CustomEvent('change', { detail: { keys } }));
  }

  fields(name) {
    return [...this.form.querySelectorAll(`[name="${name}"]`)];
  }

  coerce(name, raw) {
    return typeof this.defaults[name] === 'number' && raw !== '' && !Number.isNaN(Number(raw)) ? Number(raw) : raw;
  }

  read(name) {
    const els = this.fields(name);
    const first = els[0];
    if (!first) return this.values[name];
    if (first.hasAttribute('data-json')) {
      try { return JSON.parse(first.value); } catch { return this.values[name]; }
    }
    if (first.type === 'radio') {
      const on = els.find(e => e.checked);
      return on ? this.coerce(name, on.value) : this.values[name];
    }
    if (first.type === 'checkbox') {
      if (Array.isArray(this.defaults[name])) return els.filter(e => e.checked).map(e => e.value);
      return first.checked;
    }
    return this.coerce(name, first.value);
  }

  write() {
    for (const [name, value] of Object.entries(this.values)) {
      for (const e of this.fields(name)) {
        if (e.hasAttribute('data-json')) e.value = JSON.stringify(value);
        else if (e.type === 'radio') e.checked = String(value) === e.value;
        else if (e.type === 'checkbox') e.checked = Array.isArray(value) ? value.includes(e.value) : !!value;
        else if (e.tagName === 'SELECT') {
          if (![...e.options].some(o => o.value === String(value))) e.append(new Option(String(value), String(value)));
          e.value = String(value);
        } else if (e.type !== 'file') e.value = value;
      }
      this.showValue(name);
    }
  }

  // <output data-for="name"> shows the current value of a slider
  showValue(name) {
    for (const out of this.form.querySelectorAll(`output[data-for="${name}"]`)) out.textContent = this.values[name];
  }

  // show / hide blocks with data-show-if, and disable their inputs so they are not edited by mistake
  refresh() {
    for (const block of this.form.querySelectorAll('[data-show-if]')) {
      const ok = block.dataset.showIf.split(';').every(clause => {
        const [key, list] = clause.split(':');
        return list.split(',').includes(String(this.values[key.trim()]));
      });
      block.hidden = !ok;
    }
  }

  load() {
    try {
      const saved = JSON.parse(this.storage().getItem(this.storageKey) || '{}');
      return Object.fromEntries(Object.entries(saved).filter(([k]) => k in this.defaults));
    } catch { return {}; }
  }

  save() {
    try { this.storage().setItem(this.storageKey, JSON.stringify(this.values)); } catch { /* storage full or blocked */ }
  }
}
