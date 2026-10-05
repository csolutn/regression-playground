// A guest's login popup on the playground (partials/login_dialog.html). It logs in with POST /login as JSON,
// so the page stays: the runs made as a guest are then saved to the account (only their results, as for any
// browser run), carry(user) keeps the settings, and the page reloads as that user.
import { t } from './i18n.js';

export function mountGuestLogin(dialog, { output, toast, carry }) {
  const form = dialog.querySelector('form');
  const $ = role => form.querySelector(`[data-role="${role}"]`);
  const confirm = $('confirm'), message = $('login-message'), submit = $('login-submit');
  const { password, password_confirm: again } = form.elements;
  const say = (text, kind = 'error') => {
    message.hidden = !text;
    message.textContent = text || '';
    message.className = `flash flash-${kind}`;
  };

  for (const el of document.querySelectorAll('[data-role="login-open"]')) {
    el.addEventListener('click', e => {
      e.preventDefault();               // without this script the top bar's link goes to the login page
      if (output.run?.live) { toast(t('Wait until training finishes.'), 'info'); return; }
      say(null);
      dialog.showModal();
    });
  }
  $('login-cancel').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', e => { if (e.target === dialog && !submit.disabled) dialog.close(); });
  dialog.addEventListener('cancel', e => { if (submit.disabled) e.preventDefault(); });   // Esc while saving

  // as the login page does after each answer: a first login asks for the password again, keeping the typed
  // one only at that first ask; any other answer leaves the passwords to type again
  const answered = firstLogin => {
    const asking = firstLogin && again.disabled;          // the second password was not sent yet
    confirm.hidden = again.disabled = !firstLogin;
    password.autocomplete = firstLogin ? 'new-password' : 'current-password';
    if (!asking) password.value = '';
    again.value = '';
    (asking ? again : password).focus();
  };

  form.addEventListener('submit', async e => {
    e.preventDefault();
    submit.disabled = true;
    try {
      const res = await fetch('/login', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(Object.fromEntries(new FormData(form))),   // the second password once it is shown
      });
      let reply = null;
      try { reply = await res.json(); } catch { /* not JSON: a proxy's error page */ }
      if (!res.ok || !reply?.user) {
        say(reply?.message || t('Server error ({status})', { status: res.status }), reply?.kind);
        if (reply) answered(reply.first_login);
        submit.disabled = false;
        return;
      }
      say(t('Saving the runs you made as a guest…'), 'info');
      const failed = await output.saveGuestRuns();
      carry(reply.user);
      if (failed) alert(t('{n} of the runs made as a guest could not be saved.', { n: failed }));
      location.reload();
    } catch (err) {
      say(err.message);
      submit.disabled = false;
    }
  });
}
