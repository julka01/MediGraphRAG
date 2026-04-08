import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { BottomBar } from './components/graph/BottomBar';
import { KGPanel } from './components/kg/KGPanel';
import { Neo4jForm } from './components/kg/Neo4jForm';
import { MainLayout } from './components/layout/MainLayout';
import { Sidebar } from './components/layout/Sidebar';
import { ErrorBoundary } from './components/ui/ErrorBoundary';
import { Notifications, showSuccess } from './components/ui/Notifications';
import { useApp } from './context/AppContext';
import { useTheme } from './context/ThemeContext';
import { useModels } from './hooks/useModels';
import { useSnapToClose } from './hooks/useSnapToClose';
import { useStartup } from './hooks/useStartup';
import type { LoadNeo4jResponse, Neo4jStats } from './types/app';
import { safeSet } from './utils/storage';

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

  const rootRef = useRef<HTMLDivElement>(null);

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

  const leftSnap = useSnapToClose({
    edge: 'left',
    minSize: 288,
    onClose: () => dispatch({ type: 'CLOSE_PANEL', payload: 'left' }),
    onResize: () => {}, // fixed width, no resize — only close gesture
  });

  return (
    <div ref={rootRef} className="flex h-screen w-screen overflow-hidden">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-0 focus:left-0 focus:z-50 focus:p-2 focus:bg-base-100"
      >
        Skip to content
      </a>

      {/* Left sidebar */}
      {!state.panels.leftCollapsed && (
        <>
          <Sidebar>
            <KGPanel
              kgModelHook={kgModelHook}
              onNeo4jOpen={() => setNeo4jOpen(true)}
              onProgressStart={() => setProgressActive(true)}
              onProgressStop={() => setProgressActive(false)}
            />
          </Sidebar>
          <div
            role="separator"
            className="hidden md:block w-1 shrink-0 cursor-col-resize transition-colors bg-base-300 hover:bg-primary/50"
            onPointerDown={leftSnap.onPointerDown}
            onPointerMove={leftSnap.onPointerMove}
            onPointerUp={leftSnap.onPointerUp}
          />
        </>
      )}

      {/* Main content area */}
      <div id="main-content" className="flex flex-1 min-w-0">
        <MainLayout
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
          bottomBar={<BottomBar height={state.panels.bottomHeight} />}
        />
      </div>

      <Neo4jForm open={neo4jOpen} onClose={() => setNeo4jOpen(false)} onLoaded={handleNeo4jLoaded} />
      <Notifications />
    </div>
  );
}
