import { ArrowLeftStartOnRectangleIcon, ArrowRightStartOnRectangleIcon } from '@heroicons/react/20/solid';
import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { safeGet, safeSet } from './utils/storage';
import { OverviewPanel } from './components/graph/GraphLegend';
import { KGPanel } from './components/kg/KGPanel';
import { ModelSelector } from './components/kg/ModelSelector';
import { Neo4jForm } from './components/kg/Neo4jForm';
import { MainLayout } from './components/layout/MainLayout';
import { Sidebar } from './components/layout/Sidebar';
import { ErrorBoundary } from './components/ui/ErrorBoundary';
import { Notifications, showSuccess } from './components/ui/Notifications';
import { useApp } from './context/AppContext';
import { useTheme } from './context/ThemeContext';
import { useModels } from './hooks/useModels';
import { useStartup } from './hooks/useStartup';
import type { LoadNeo4jResponse, Neo4jStats } from './types/app';

const GraphContainer = lazy(() =>
  import('./components/graph/GraphContainer').then((m) => ({ default: m.GraphContainer })),
);
const ChatPanel = lazy(() => import('./components/chat/ChatPanel').then((m) => ({ default: m.ChatPanel })));

function PanelSkeleton() {
  return (
    <div className="flex flex-col h-full animate-pulse p-4">
      <div className="h-6 w-40 bg-base-300 rounded mb-4" />
      <div className="flex-1 bg-base-300/30 rounded" />
    </div>
  );
}

export default function App() {
  const { state, dispatch } = useApp();
  const { theme } = useTheme();

  const startup = useStartup(['openai', 'openrouter']);
  const kgModelHook = useModels('openai', startup.modelsByVendor.openai);
  const ragModelHook = useModels('openrouter', startup.modelsByVendor.openrouter);

  const [neo4jOpen, setNeo4jOpen] = useState(false);
  const [progressActive, setProgressActive] = useState(false);

  const SIDEBAR_STORAGE_KEY = 'sidebar-width';
  const DEFAULT_SIDEBAR_WIDTH = 288; // 18rem = w-72
  const MIN_SIDEBAR_WIDTH = 200;
  const MAX_SIDEBAR_WIDTH = 480;

  const [sidebarWidth, setSidebarWidth] = useState(() => {
    const stored = safeGet(SIDEBAR_STORAGE_KEY);
    if (!stored) return DEFAULT_SIDEBAR_WIDTH;
    const parsed = Number(stored);
    return Number.isFinite(parsed) ? parsed : DEFAULT_SIDEBAR_WIDTH;
  });
  const [sidebarDragging, setSidebarDragging] = useState(false);
  const sidebarHandleRef = useRef<HTMLDivElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  const onSidebarPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    sidebarHandleRef.current?.setPointerCapture(e.pointerId);
    setSidebarDragging(true);
    rootRef.current?.classList.add('select-none');
  }, []);

  const onSidebarPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!sidebarDragging) return;
      const root = rootRef.current;
      if (!root) return;
      const x = e.clientX - root.getBoundingClientRect().left;
      const clamped = Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, x));
      setSidebarWidth(clamped);
      safeSet(SIDEBAR_STORAGE_KEY, String(Math.round(clamped)));
    },
    [sidebarDragging],
  );

  const onSidebarPointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    sidebarHandleRef.current?.releasePointerCapture(e.pointerId);
    setSidebarDragging(false);
    rootRef.current?.classList.remove('select-none');
  }, []);

  const resetSidebarWidth = useCallback(() => {
    setSidebarWidth(DEFAULT_SIDEBAR_WIDTH);
    safeSet(SIDEBAR_STORAGE_KEY, String(DEFAULT_SIDEBAR_WIDTH));
  }, []);

  useEffect(() => {
    if (!startup.loading && startup.kgList.length > 0) {
      dispatch({ type: 'SET_KG_LIST', kgList: startup.kgList });
    }
  }, [startup.loading, startup.kgList, dispatch]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        const textarea = document.querySelector<HTMLTextAreaElement>('textarea[placeholder="Ask a question…"]');
        if (textarea) {
          textarea.focus();
          textarea.select();
        }
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleNeo4jLoaded = useCallback(
    (result: LoadNeo4jResponse, kgFilter: string, stats: Neo4jStats) => {
      dispatch({ type: 'SET_KG', kgId: result.kg_id, kgName: result.kg_name || kgFilter || null });
      if (result.kg_name) safeSet('currentKGName', result.kg_name);
      dispatch({ type: 'SET_GRAPH_DATA', data: result.graph_data });
      dispatch({ type: 'CLEAR_HIGHLIGHTED_NODES' });
      dispatch({ type: 'CLEAR_FILTERS' });

      const nodeCount = result.graph_data?.nodes?.length ?? 0;
      const relCount = result.graph_data?.relationships?.length ?? 0;
      let msg = result.message || `Loaded KG from Neo4j with ${nodeCount} nodes and ${relCount} relationships`;
      if (stats?.sample_mode) msg += ' (Smart Sample)';
      else if (stats?.complete_import) msg += ' (Complete Import)';
      showSuccess(dispatch, msg);
    },
    [dispatch],
  );

  void theme; // used by ThemeContext to apply data-theme attribute

  return (
    <div ref={rootRef} className="flex h-screen w-screen overflow-hidden">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-0 focus:left-0 focus:z-50 focus:p-2 focus:bg-base-100"
      >
        Skip to content
      </a>
      <Sidebar width={sidebarWidth}>
        <Sidebar.Header healthData={startup.health} />
        <Sidebar.Section title="Knowledge Graph">
          <ModelSelector label="KG" vendorHook={kgModelHook} />
          <div className="mt-2">
            <KGPanel
              kgModelHook={kgModelHook}
              onNeo4jOpen={() => setNeo4jOpen(true)}
              onProgressStart={() => setProgressActive(true)}
              onProgressStop={() => setProgressActive(false)}
            />
          </div>
        </Sidebar.Section>
        <Sidebar.Section title="RAG Model">
          <ModelSelector label="RAG" vendorHook={ragModelHook} />
        </Sidebar.Section>
        <OverviewPanel />
      </Sidebar>
      <div className="hidden md:flex items-center justify-center shrink-0">
        <button
          type="button"
          className="btn btn-ghost btn-xs btn-square"
          onClick={() => dispatch({ type: 'TOGGLE_SIDEBAR' })}
          title={state.sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-label={state.sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {state.sidebarCollapsed ? (
            <ArrowRightStartOnRectangleIcon className="size-4" aria-hidden="true" />
          ) : (
            <ArrowLeftStartOnRectangleIcon className="size-4" aria-hidden="true" />
          )}
        </button>
      </div>
      {/* biome-ignore lint/a11y/useSemanticElements: <hr> cannot hold pointer event handlers or refs needed for drag behavior */}
      <div
        ref={sidebarHandleRef}
        role="separator"
        tabIndex={0}
        aria-orientation="vertical"
        aria-valuenow={Math.round(sidebarWidth)}
        className={`hidden md:block w-1 shrink-0 cursor-col-resize transition-colors ${
          sidebarDragging ? 'bg-primary/50' : 'bg-base-300 hover:bg-primary/50'
        }`}
        onPointerDown={onSidebarPointerDown}
        onPointerMove={onSidebarPointerMove}
        onPointerUp={onSidebarPointerUp}
        onDoubleClick={resetSidebarWidth}
      />

      <div id="main-content" className="flex flex-1 min-w-0">
        <MainLayout
          layout={state.layout ?? (state.kgExpanded ? 'graph-only' : 'split')}
          graphPanel={
            <ErrorBoundary name="Knowledge Graph">
              <Suspense fallback={<PanelSkeleton />}>
                <GraphContainer progressActive={progressActive} onProgressClose={() => setProgressActive(false)} />
              </Suspense>
            </ErrorBoundary>
          }
          chatPanel={
            <ErrorBoundary name="RAG Chat">
              <Suspense fallback={<PanelSkeleton />}>
                <ChatPanel ragModelHook={ragModelHook} />
              </Suspense>
            </ErrorBoundary>
          }
        />
      </div>

      <Neo4jForm open={neo4jOpen} onClose={() => setNeo4jOpen(false)} onLoaded={handleNeo4jLoaded} />
      <Notifications />
    </div>
  );
}
