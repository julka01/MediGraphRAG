import { createContext, useContext, useReducer, useRef } from 'react';
import type { AppAction, AppContextValue, AppState, Layout, PanelState, ViewState } from '../types/app';

const AppContext = createContext<AppContextValue | null>(null);

const PANEL_STATE_KEY = 'panel-state';

const initialPanelState: PanelState = {
  leftCollapsed: false,
  rightCollapsed: false,
  bottomCollapsed: false,
  rightWidth: 320,
  bottomHeight: 120,
};

function loadPanelState(): PanelState {
  try {
    const stored = localStorage.getItem(PANEL_STATE_KEY);
    if (stored) {
      return { ...initialPanelState, ...JSON.parse(stored) };
    }
  } catch {
    // Ignore parse errors and fall back to defaults
  }
  return initialPanelState;
}

function savePanelState(panels: PanelState): void {
  try {
    localStorage.setItem(PANEL_STATE_KEY, JSON.stringify(panels));
  } catch {
    // Ignore storage errors (e.g. private browsing quota exceeded)
  }
}

const initialState: AppState = {
  currentKGId: null,
  currentKGName: null,
  kgList: [],
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
  activeView: 'kg',
  layout: 'split' as Layout,
  panels: loadPanelState(),
  kgExpanded: false,
  notification: null,
};

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_KG':
      return { ...state, currentKGId: action.kgId, currentKGName: action.kgName };
    case 'SET_KG_LIST':
      return { ...state, kgList: action.kgList };
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
    case 'SET_FILTERS': {
      return {
        ...state,
        currentFilters: {
          nodeTypes: new Set(action.nodeTypes ?? state.currentFilters.nodeTypes),
          relationshipTypes: new Set(action.relationshipTypes ?? state.currentFilters.relationshipTypes),
        },
      };
    }
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
    case 'SET_VIEW':
      return { ...state, activeView: action.view };
    // TOGGLE_SIDEBAR kept for backward compatibility — maps to left panel
    case 'TOGGLE_SIDEBAR':
    case 'TOGGLE_LEFT_PANEL': {
      const panels = { ...state.panels, leftCollapsed: !state.panels.leftCollapsed };
      savePanelState(panels);
      return { ...state, panels };
    }
    case 'TOGGLE_RIGHT_PANEL': {
      const panels = { ...state.panels, rightCollapsed: !state.panels.rightCollapsed };
      savePanelState(panels);
      return { ...state, panels };
    }
    case 'TOGGLE_BOTTOM_PANEL': {
      const panels = { ...state.panels, bottomCollapsed: !state.panels.bottomCollapsed };
      savePanelState(panels);
      return { ...state, panels };
    }
    case 'SET_RIGHT_WIDTH': {
      const panels = { ...state.panels, rightWidth: action.payload };
      savePanelState(panels);
      return { ...state, panels };
    }
    case 'SET_BOTTOM_HEIGHT': {
      const panels = { ...state.panels, bottomHeight: action.payload };
      savePanelState(panels);
      return { ...state, panels };
    }
    case 'CLOSE_PANEL': {
      const key = ({ left: 'leftCollapsed', right: 'rightCollapsed', bottom: 'bottomCollapsed' } as const)[action.payload];
      const panels = { ...state.panels, [key]: true };
      savePanelState(panels);
      return { ...state, panels };
    }
    case 'OPEN_PANEL': {
      const key = ({ left: 'leftCollapsed', right: 'rightCollapsed', bottom: 'bottomCollapsed' } as const)[action.payload];
      const panels = { ...state.panels, [key]: false };
      savePanelState(panels);
      return { ...state, panels };
    }
    case 'TOGGLE_KG_EXPANDED': {
      const newLayout = state.layout === 'graph-only' ? 'split' : 'graph-only';
      return { ...state, kgExpanded: !state.kgExpanded, layout: newLayout };
    }
    case 'SET_LAYOUT':
      return { ...state, layout: action.layout, kgExpanded: action.layout === 'graph-only' };
    case 'SET_CLUSTERS':
      return { ...state, clusters: action.clusters };
    case 'SHOW_NOTIFICATION':
      return { ...state, notification: { type: action.notifType, message: action.message } };
    case 'CLEAR_NOTIFICATION':
      return { ...state, notification: null };
    case 'CLEAR_KG':
      return {
        ...state,
        currentKGId: null,
        currentKGName: null,
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

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);
  const networkRef = useRef<vis.Network | null>(null);
  const idCounterRef = useRef<number>(0);
  const initialViewRef = useRef<ViewState | null>(null);

  return (
    <AppContext.Provider value={{ state, dispatch, networkRef, idCounterRef, initialViewRef }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
}
