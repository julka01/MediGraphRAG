import { state } from '../state.js';

export function getGraphTheme() {
  const dark = document.body.dataset.theme === 'dark';
  return {
    nodeText:       dark ? '#ffffff' : '#1a1a1a',
    nodeTextDimmed: dark ? '#444444' : '#bbbbbb',
    edgeText:       dark ? '#888888' : '#555555',
    edgeLabelBg:    dark ? 'rgba(10,10,10,0.80)' : 'rgba(232,236,240,0.88)',
    dimmedNodeBg:   dark ? '#2a2a2a' : '#d8dde4',
    dimmedNodeBdr:  dark ? '#3a3a3a' : '#c8cdd4',
    dimmedEdge:     dark ? '#282828' : '#dde0e4',
  };
}

export function initTheme() {
  const saved = localStorage.getItem('kg-theme') || 'dark';
  document.body.dataset.theme = saved;
  document.getElementById('theme-icon').textContent = saved === 'dark' ? '🌙' : '☀️';

  document.getElementById('theme-toggle').addEventListener('click', function () {
    const next = document.body.dataset.theme === 'dark' ? 'light' : 'dark';
    document.body.dataset.theme = next;
    localStorage.setItem('kg-theme', next);
    document.getElementById('theme-icon').textContent = next === 'dark' ? '🌙' : '☀️';
    // Dynamic import avoids circular dependency: theme → network → legend → theme
    if (state.graphData) {
      import('../graph/network.js').then(({ initializeGraph }) => {
        initializeGraph(state.graphData);
      });
    }
  });
}
