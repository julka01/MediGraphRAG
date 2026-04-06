import { api } from '../api.js';

export function initHealth() {
  const dot = document.getElementById('doctor-dot');
  if (!dot) return;

  async function checkHealth() {
    try {
      const d = await api.checkHealth();
      const colors = { ok: '#2ecc71', warn: '#f39c12', fail: '#e74c3c' };
      dot.style.background = colors[d.status] || '#888';
      const failed = (d.checks || []).filter(c => c.status === 'fail').map(c => c.check).join(', ');
      const warned = (d.checks || []).filter(c => c.status === 'warn').map(c => c.check).join(', ');
      let tip = `System: ${d.status.toUpperCase()}`;
      if (failed) tip += `\nFailed: ${failed}`;
      if (warned) tip += `\nWarnings: ${warned}`;
      dot.title = tip;
    } catch (e) {
      dot.style.background = '#e74c3c';
      dot.title = 'Health check failed — server may be unreachable';
    }
  }

  checkHealth();

  dot.addEventListener('click', async () => {
    dot.style.background = 'var(--text-3)';
    dot.title = 'Rechecking…';
    try {
      const d = await api.checkHealth();
      const colors = { ok: '#2ecc71', warn: '#f39c12', fail: '#e74c3c' };
      dot.style.background = colors[d.status] || '#888';
      const failed = (d.checks || []).filter(c => c.status === 'fail').map(c => c.check).join(', ');
      const warned = (d.checks || []).filter(c => c.status === 'warn').map(c => c.check).join(', ');
      let tip = `System: ${d.status.toUpperCase()}`;
      if (failed) tip += `\nFailed: ${failed}`;
      if (warned) tip += `\nWarnings: ${warned}`;
      dot.title = tip;
    } catch (e) {
      dot.style.background = '#e74c3c';
      dot.title = 'Health check failed';
    }
  });
}
