// Shared mutable application state.
// All modules import { state } and read/write properties directly.
export const state = {
  network: null,
  currentKGId: null,
  currentKGName: null,
  graphData: null,
  fullGraphData: null,
  highlightedNodes: new Set(),
  uniqueIdCounter: 0,
  nodeTypeColors: {},
  relationshipTypeColors: {},
  currentFilters: { nodeTypes: new Set(), relationshipTypes: new Set() },
  clusters: {},
  initialViewState: null,
  physicsEnabled: true,
  nodeSizeMetric: 'fixed',
  showEdgeLabels: true,
};

// Clear stale KG name from previous session on load
localStorage.removeItem('currentKGName');
