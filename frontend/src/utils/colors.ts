const NODE_PALETTE_FALLBACKS: readonly string[] = [
  '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
  '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928',
  '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
  '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f',
];

const EDGE_PALETTE_FALLBACKS: readonly string[] = [
  '#444444', '#666666', '#888888', '#aaaaaa', '#cccccc',
  '#2c3e50', '#34495e', '#7f8c8d', '#95a5a6', '#bdc3c7',
];

export function readCSSColor(property: string, fallback: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(property).trim();
  return value || fallback;
}

function getNodePalette(): readonly string[] {
  return NODE_PALETTE_FALLBACKS.map((fallback, i) =>
    readCSSColor(`--color-graph-node-${i}`, fallback),
  );
}

function getEdgePalette(): readonly string[] {
  return EDGE_PALETTE_FALLBACKS.map((fallback, i) =>
    readCSSColor(`--color-graph-edge-${i}`, fallback),
  );
}

export function generateNodeTypeColors(nodeTypes: string[]): Record<string, string> {
  const palette = getNodePalette();
  const colorMap: Record<string, string> = {};
  nodeTypes.forEach((type, index) => {
    colorMap[type] = palette[index % palette.length];
  });
  return colorMap;
}

export function generateRelationshipTypeColors(relationshipTypes: string[]): Record<string, string> {
  const palette = getEdgePalette();
  const colorMap: Record<string, string> = {};
  relationshipTypes.forEach((type, index) => {
    colorMap[type] = palette[index % palette.length];
  });
  return colorMap;
}
