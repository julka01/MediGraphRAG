const NODE_PALETTE = [
  '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
  '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928',
  '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
  '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f',
];

const EDGE_PALETTE = [
  '#444444', '#666666', '#888888', '#aaaaaa', '#cccccc',
  '#2c3e50', '#34495e', '#7f8c8d', '#95a5a6', '#bdc3c7',
];

export function generateNodeTypeColors(nodeTypes) {
  const colorMap = {};
  nodeTypes.forEach((type, index) => {
    colorMap[type] = NODE_PALETTE[index % NODE_PALETTE.length];
  });
  return colorMap;
}

export function generateRelationshipTypeColors(relationshipTypes) {
  const colorMap = {};
  relationshipTypes.forEach((type, index) => {
    colorMap[type] = EDGE_PALETTE[index % EDGE_PALETTE.length];
  });
  return colorMap;
}
