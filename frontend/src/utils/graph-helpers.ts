import { readCSSColor } from './colors';

export function confidenceEdgeColor(conf: number | null | undefined): string | null {
  if (conf == null) return null;
  if (conf >= 0.8) return readCSSColor('--color-success', '#f1c40f');
  if (conf >= 0.5) return readCSSColor('--color-warning', '#e67e22');
  if (conf >= 0.3) return readCSSColor('--color-neutral', '#95a5a6');
  return readCSSColor('--color-base-300', '#555e68');
}

export interface GraphTheme {
  nodeText: string;
  nodeTextDimmed: string;
  edgeText: string;
  edgeLabelBg: string;
  dimmedNodeBg: string;
  dimmedNodeBdr: string;
  dimmedEdge: string;
  highlight: string;
}

export function getGraphTheme(): GraphTheme {
  return {
    nodeText: readCSSColor('--color-graph-node-text', '#ffffff'),
    nodeTextDimmed: readCSSColor('--color-graph-node-text-dimmed', '#444444'),
    edgeText: readCSSColor('--color-graph-edge-text', '#888888'),
    edgeLabelBg: readCSSColor('--color-graph-edge-label-bg', 'transparent'),
    dimmedNodeBg: readCSSColor('--color-graph-dimmed-bg', '#2a2a2a'),
    dimmedNodeBdr: readCSSColor('--color-graph-dimmed-border', '#3a3a3a'),
    dimmedEdge: readCSSColor('--color-graph-dimmed-edge', '#282828'),
    highlight: readCSSColor('--color-graph-highlight', '#ffd700'),
  };
}

export function normName(s: string | undefined | null): string {
  return (s || '')
    .toLowerCase()
    .replace(/[_\s]+/g, ' ')
    .trim();
}
