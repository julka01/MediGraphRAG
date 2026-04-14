/* Theme */
export type Theme = 'light' | 'dark';

export interface ThemeContextValue {
  theme: Theme;
  toggleTheme: () => void;
}

/* Views & metrics */
export type ActiveView = 'kg' | 'chat';
export type Layout = 'split' | 'graph-only' | 'chat-only';
export type NodeSizeMetric = 'uniform' | 'degree' | 'inDegree' | 'outDegree' | 'pageRank';
export type NotificationType = 'error' | 'success';

/* Notification */
export interface Notification {
  type: NotificationType;
  message: string;
}

/* Filters */
export interface Filters {
  /** null = uninitialized (show all); empty Set = user explicitly cleared all */
  nodeTypes: Set<string> | null;
  relationshipTypes: Set<string> | null;
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

/* Panel state */
export interface PanelState {
  leftCollapsed: boolean;
  rightCollapsed: boolean;
  bottomCollapsed: boolean;
  topCollapsed: boolean;
  rightWidth: number;
  bottomHeight: number;
}

/* App state */
export interface AppState {
  currentKGId: string | null;
  currentKGName: string | null;
  kgList: KGListItem[];
  graphData: GraphData | null;
  fullGraphData: GraphData | null;
  highlightedNodes: Set<string>;
  highlightedCount: number;
  searchTerm: string;
  nodeTypeColors: Record<string, string>;
  relationshipTypeColors: Record<string, string>;
  currentFilters: Filters;
  clusters: Record<string, unknown>;
  physicsEnabled: boolean;
  layoutSpacing: number;
  nodeSizeMetric: NodeSizeMetric;
  showEdgeLabels: boolean;
  activeView: ActiveView;
  layout: Layout;
  panels: PanelState;
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
  | { type: 'SET_HIGHLIGHTED_COUNT'; count: number }
  | { type: 'SET_NODE_TYPE_COLORS'; colors: Record<string, string> }
  | { type: 'SET_RELATIONSHIP_TYPE_COLORS'; colors: Record<string, string> }
  | { type: 'SET_FILTERS'; nodeTypes?: Iterable<string>; relationshipTypes?: Iterable<string> }
  | { type: 'CLEAR_FILTERS' }
  | { type: 'TOGGLE_PHYSICS' }
  | { type: 'SET_PHYSICS'; enabled: boolean }
  | { type: 'SET_LAYOUT_SPACING'; spacing: number }
  | { type: 'SET_NODE_SIZE_METRIC'; metric: NodeSizeMetric }
  | { type: 'TOGGLE_EDGE_LABELS' }
  | { type: 'SET_VIEW'; view: ActiveView }
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'TOGGLE_LEFT_PANEL' }
  | { type: 'TOGGLE_RIGHT_PANEL' }
  | { type: 'TOGGLE_BOTTOM_PANEL' }
  | { type: 'TOGGLE_TOP_PANEL' }
  | { type: 'SET_RIGHT_WIDTH'; payload: number }
  | { type: 'SET_BOTTOM_HEIGHT'; payload: number }
  | { type: 'CLOSE_PANEL'; payload: 'left' | 'right' | 'bottom' | 'top' }
  | { type: 'OPEN_PANEL'; payload: 'left' | 'right' | 'bottom' | 'top' }
  | { type: 'SET_SEARCH_TERM'; payload: string }
  | { type: 'TOGGLE_KG_EXPANDED' }
  | { type: 'SET_LAYOUT'; layout: Layout }
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
  reasoningEdges?: ReasoningEdge[];
  sourceEntities?: SourceEntity[];
  entityNames?: Set<string>;
  trustSignals?: TrustSignals;
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

export interface ChatPayload {
  question: string;
  provider_rag: string;
  model_rag: string;
  kg_name?: string;
  dataset_name?: string;
  task_type?: string;
  runtime_guardrail?: boolean;
  runtime_guardrail_mode?: 'retry_then_abstain' | 'abstain_only';
}

export interface ChatRequestOptions {
  datasetName?: string;
  taskType?: string;
  runtimeGuardrail?: boolean;
  runtimeGuardrailMode?: 'retry_then_abstain' | 'abstain_only';
}

export interface TrustSignals {
  structural_support?: number | null;
  grounding_support?: number | null;
  confidence?: number | null;
}

export interface ChatResponse {
  response?: string;
  message?: string;
  info?: {
    confidence_score?: number;
    structural_support?: number | null;
    grounding_support?: number | null;
    confidence?: number | null;
    kg_confidence?: number | null;
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

export interface KGSettings {
  provider?: string | null;
  model?: string | null;
  embeddingModel?: string | null;
  maxChunks?: number | null;
}

export interface LoadNeo4jResponse {
  kg_id: string;
  kg_name?: string;
  message?: string;
  graph_data: GraphData;
  stats?: Neo4jStats;
  kg_settings?: KGSettings | null;
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
  restoreVendorModel: (vendor: string, model: string) => Promise<void>;
}

export interface UseHealthReturn {
  status: { level: string; tip: string };
  checking: boolean;
  checkHealth: () => Promise<void>;
}

export interface UseChatReturn {
  messages: ChatMessage[];
  sending: boolean;
  addMessage: (msg: ChatMessage) => void;
  sendQuestion: (
    question: string,
    kgName: string | null,
    vendor: string,
    model: string,
    options?: ChatRequestOptions,
  ) => Promise<ChatResponse>;
  clearChat: () => void;
  exportChat: (kgName: string | null) => void;
}
