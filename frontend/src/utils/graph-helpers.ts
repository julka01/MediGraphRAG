export function confidenceEdgeColor(conf) {
  if (conf == null) return null;
  if (conf >= 0.8) return '#f1c40f';
  if (conf >= 0.5) return '#e67e22';
  if (conf >= 0.3) return '#95a5a6';
  return '#555e68';
}

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

export function normName(s) {
  return (s || '').toLowerCase().replace(/[_\s]+/g, ' ').trim();
}
