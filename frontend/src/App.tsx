import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { BottomBar } from './components/graph/BottomBar';
import { DatabaseSettings } from './components/kg/DatabaseSettings';
import { KGPanel } from './components/kg/KGPanel';
import { MainLayout } from './components/layout/MainLayout';
import { Sidebar } from './components/layout/Sidebar';
import { ErrorBoundary } from './components/ui/ErrorBoundary';
import { Notifications, showError, showSuccess } from './components/ui/Notifications';
import { useApp } from './context/AppContext';
import { useTheme } from './context/ThemeContext';
import { useModels } from './hooks/useModels';
import { useSnapToClose } from './hooks/useSnapToClose';
import { useStartup } from './hooks/useStartup';
import { api } from './api';
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
  const ragModelHook = useModels('openai', startup.modelsByVendor.openai);

  const [settingsOpen, setSettingsOpen] = useState(false);
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

  const handleLoadKG = useCallback(
    async (loadMode: string, nodeLimit: number, kgFilter: string) => {
      const credStr = sessionStorage.getItem('neo4j-credentials');
      if (!credStr) {
        showError(dispatch, 'Please configure database settings first');
        return;
      }
      let creds: { uri: string; user: string; password: string };
      try {
        creds = JSON.parse(credStr) as { uri: string; user: string; password: string };
      } catch {
        showError(dispatch, 'Saved credentials are corrupted. Please reconfigure in Database Settings.');
        return;
      }

      const formData = new FormData();
      formData.append('uri', creds.uri);
      formData.append('user', creds.user);
      formData.append('password', creds.password);
      if (kgFilter) formData.append('kg_label', kgFilter);

      switch (loadMode) {
        case 'limited':
          formData.append('limit', String(nodeLimit));
          formData.append('sample_mode', 'false');
          formData.append('load_complete', 'false');
          break;
        case 'sample':
          formData.append('sample_mode', 'true');
          formData.append('load_complete', 'false');
          if (nodeLimit) formData.append('limit', String(nodeLimit));
          break;
        case 'complete':
          formData.append('load_complete', 'true');
          formData.append('sample_mode', 'false');
          break;
      }

      try {
        const result = await api.loadFromNeo4j(formData);
        handleNeo4jLoaded(result, kgFilter, result.stats ?? {} as Neo4jStats);
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        showError(dispatch, `Loading failed: ${msg}`);
      }
    },
    [dispatch, handleNeo4jLoaded],
  );

  void theme; // used by ThemeContext to apply data-theme attribute

  const leftSnap = useSnapToClose({
    edge: 'left',
    minSize: 288,
    onClose: () => dispatch({ type: 'CLOSE_PANEL', payload: 'left' }),
    onOpen: () => dispatch({ type: 'OPEN_PANEL', payload: 'left' }),
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
          <Sidebar onSettingsClick={() => setSettingsOpen(true)}>
            <KGPanel
              kgModelHook={kgModelHook}
              onLoadKG={handleLoadKG}
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

      <DatabaseSettings
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
      <Notifications />
    </div>
  );
}
