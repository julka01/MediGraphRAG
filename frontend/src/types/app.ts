/* Theme */
export type Theme = 'light' | 'dark';

export interface ThemeContextValue {
  theme: Theme;
  toggleTheme: () => void;
}

/* Views & metrics */
export type ActiveView = 'kg' | 'chat';
export type NodeSizeMetric = 'fixed' | 'degree' | 'betweenness';
export type NotificationType = 'error' | 'success';

/* Notification */
export interface Notification {
  type: NotificationType;
  message: string;
}

/* Filters */
export interface Filters {
  nodeTypes: Set<string>;
  relationshipTypes: Set<string>;
}

/* Graph data from API */
export interface GraphNode {
  id: string | number;
  labels?: string[];
  properties?: Record<string, unknown>;
}

export interface GraphRelationship {
  id?: string | number;
  type?: string;
  from?: string | number;
  start?: string | number;
  source?: string | number;
  to?: string | number;
  end?: string | number;
  target?: string | number;
  properties?: Record<string, unknown>;
}

export interface GraphData {
  nodes: GraphNode[];
  relationships: GraphRelationship[];
}

/* App state */
export interface AppState {
  currentKGId: string | null;
  currentKGName: string | null;
  kgList: KGListItem[];
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
  activeView: ActiveView;
  sidebarCollapsed: boolean;
  kgExpanded: boolean;
  notification: Notification | null;
}

/* KG list item from API */
export interface KGListItem {
  name: string;
}

/* Reducer actions */
export type AppAction =
  | { type: 'SET_KG'; kgId: string; kgName: string | null }
  | { type: 'SET_KG_LIST'; kgList: KGListItem[] }
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
  | { type: 'SET_VIEW'; view: ActiveView }
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'TOGGLE_KG_EXPANDED' }
  | { type: 'SET_CLUSTERS'; clusters: Record<string, unknown> }
  | { type: 'SHOW_NOTIFICATION'; notifType: NotificationType; message: string }
  | { type: 'CLEAR_NOTIFICATION' }
  | { type: 'CLEAR_KG' };

/* Context value */
export interface ViewState {
  scale: number;
  position: { x: number; y: number };
}

export interface AppContextValue {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
  networkRef: React.MutableRefObject<vis.Network | null>;
  idCounterRef: React.MutableRefObject<number>;
  initialViewRef: React.MutableRefObject<ViewState | null>;
}

/* Chat */
export interface ChatMessage {
  type: 'user' | 'ai' | 'thinking' | 'error';
  message: string;
  ts?: number;
  sections?: ResponseSections;
  sourceChip?: string;
  reasoningEdges?: ReasoningEdge[];
  sourceEntities?: SourceEntity[];
}

export interface ResponseSections {
  recommendation: string;
  reasoning: string;
  evidence: string;
  nextSteps: string;
  fallback: string;
}

export interface ReasoningEdge {
  from?: string;
  from_name?: string;
  to?: string;
  to_name?: string;
  relationship?: string;
}

export interface SourceEntity {
  id?: string;
  description?: string;
}

/* API response types */
export interface HealthCheck {
  check: string;
  status: 'ok' | 'warn' | 'fail';
}

export interface HealthResponse {
  status: string;
  checks?: HealthCheck[];
}

export interface ModelsResponse {
  models: string[];
}

export interface KGListResponse {
  kgs: KGListItem[];
}

export interface DefaultCredentialsResponse {
  uri?: string;
  user?: string;
}

export interface ChatPayload {
  question: string;
  provider_rag: string;
  model_rag: string;
  kg_name?: string;
}

export interface ChatResponse {
  response?: string;
  message?: string;
  info?: {
    confidence_score?: number;
    entities?: {
      used_entities?: SourceEntity[];
      reasoning_edges?: ReasoningEdge[];
    };
  };
}

export interface CreateKGResponse {
  kg_id: string;
  kg_name?: string;
  graph_data?: GraphData;
}

export interface LoadNeo4jResponse {
  kg_id: string;
  kg_name?: string;
  message?: string;
  graph_data: GraphData;
  stats?: Neo4jStats;
}

export interface Neo4jStats {
  total_nodes_in_db?: number;
  total_relationships_in_db?: number;
  loaded_nodes?: number;
  loaded_relationships?: number;
  sample_mode?: boolean;
  complete_import?: boolean;
}

export interface ClearKGResponse {
  message?: string;
}

/* Hook return types */
export interface UseModelsReturn {
  vendor: string;
  models: string[];
  selectedModel: string;
  loading: boolean;
  setSelectedModel: (model: string) => void;
  changeVendor: (vendor: string) => void;
  fetchModels: (vendor?: string) => Promise<void>;
}

export interface UseHealthReturn {
  status: { color: string; tip: string };
  checking: boolean;
  checkHealth: () => Promise<void>;
}

export interface UseChatReturn {
  messages: ChatMessage[];
  sending: boolean;
  addMessage: (msg: ChatMessage) => void;
  sendQuestion: (question: string, kgName: string | null, vendor: string, model: string) => Promise<ChatResponse>;
  clearChat: () => void;
  exportChat: (kgName: string | null) => void;
}
