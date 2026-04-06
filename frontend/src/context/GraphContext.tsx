import { createContext, useContext, useReducer, useRef } from 'react';
import type { Filters, GraphData, ViewState } from '../types/app';

interface GraphState {
  graphData: GraphData | null;
  fullGraphData: GraphData | null;
  highlightedNodes: Set<string>;
  nodeTypeColors: Record<string, string>;
  relationshipTypeColors: Record<string, string>;
  currentFilters: Filters;
  clusters: Record<string, unknown>;
  physicsEnabled: boolean;
  nodeSizeMetric: string;
  showEdgeLabels: boolean;
}

type GraphAction =
  | { type: 'SET_GRAPH_DATA'; data: GraphData | null }
  | { type: 'SET_FULL_GRAPH_DATA'; data: GraphData | null }
  | { type: 'SET_HIGHLIGHTED_NODES'; nodes: Iterable<string> }
  | { type: 'CLEAR_HIGHLIGHTED_NODES' }
  | { type: 'SET_NODE_TYPE_COLORS'; colors: Record<string, string> }
  | { type: 'SET_RELATIONSHIP_TYPE_COLORS'; colors: Record<string, string> }
  | { type: 'SET_FILTERS'; nodeTypes?: Iterable<string>; relationshipTypes?: Iterable<string> }
  | { type: 'CLEAR_FILTERS' }
  | { type: 'TOGGLE_PHYSICS' }
  | { type: 'SET_PHYSICS'; enabled: boolean }
  | { type: 'SET_NODE_SIZE_METRIC'; metric: string }
  | { type: 'TOGGLE_EDGE_LABELS' }
  | { type: 'SET_CLUSTERS'; clusters: Record<string, unknown> }
  | { type: 'CLEAR_GRAPH' };

interface GraphContextValue {
  state: GraphState;
  dispatch: React.Dispatch<GraphAction>;
  networkRef: React.RefObject<vis.Network | null>;
  idCounterRef: React.RefObject<number>;
  initialViewRef: React.RefObject<ViewState | null>;
}

const GraphContext = createContext<GraphContextValue | null>(null);

const initialState: GraphState = {
  graphData: null,
  fullGraphData: null,
  highlightedNodes: new Set(),
  nodeTypeColors: {},
  relationshipTypeColors: {},
  currentFilters: { nodeTypes: new Set(), relationshipTypes: new Set() },
  clusters: {},
  physicsEnabled: true,
  nodeSizeMetric: 'fixed',
  showEdgeLabels: true,
};

function graphReducer(state: GraphState, action: GraphAction): GraphState {
  switch (action.type) {
    case 'SET_GRAPH_DATA':
      return { ...state, graphData: action.data };
    case 'SET_FULL_GRAPH_DATA':
      return { ...state, fullGraphData: action.data };
    case 'SET_HIGHLIGHTED_NODES':
      return { ...state, highlightedNodes: new Set(action.nodes) };
    case 'CLEAR_HIGHLIGHTED_NODES':
      return { ...state, highlightedNodes: new Set() };
    case 'SET_NODE_TYPE_COLORS':
      return { ...state, nodeTypeColors: action.colors };
    case 'SET_RELATIONSHIP_TYPE_COLORS':
      return { ...state, relationshipTypeColors: action.colors };
    case 'SET_FILTERS':
      return {
        ...state,
        currentFilters: {
          nodeTypes: new Set(action.nodeTypes ?? state.currentFilters.nodeTypes),
          relationshipTypes: new Set(action.relationshipTypes ?? state.currentFilters.relationshipTypes),
        },
      };
    case 'CLEAR_FILTERS':
      return { ...state, currentFilters: { nodeTypes: new Set(), relationshipTypes: new Set() } };
    case 'TOGGLE_PHYSICS':
      return { ...state, physicsEnabled: !state.physicsEnabled };
    case 'SET_PHYSICS':
      return { ...state, physicsEnabled: action.enabled };
    case 'SET_NODE_SIZE_METRIC':
      return { ...state, nodeSizeMetric: action.metric };
    case 'TOGGLE_EDGE_LABELS':
      return { ...state, showEdgeLabels: !state.showEdgeLabels };
    case 'SET_CLUSTERS':
      return { ...state, clusters: action.clusters };
    case 'CLEAR_GRAPH':
      return {
        ...state,
        graphData: null,
        fullGraphData: null,
        highlightedNodes: new Set(),
        nodeTypeColors: {},
        relationshipTypeColors: {},
        currentFilters: { nodeTypes: new Set(), relationshipTypes: new Set() },
        clusters: {},
      };
    default:
      return state;
  }
}

export function GraphProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(graphReducer, initialState);
  const networkRef = useRef<vis.Network | null>(null);
  const idCounterRef = useRef<number>(0);
  const initialViewRef = useRef<ViewState | null>(null);

  return (
    <GraphContext.Provider value={{ state, dispatch, networkRef, idCounterRef, initialViewRef }}>
      {children}
    </GraphContext.Provider>
  );
}

export function useGraphState(): GraphContextValue {
  const ctx = useContext(GraphContext);
  if (!ctx) throw new Error('useGraphState must be used within GraphProvider');
  return ctx;
}

export type { GraphAction, GraphState };
