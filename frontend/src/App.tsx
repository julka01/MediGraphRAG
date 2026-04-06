import { lazy, Suspense, useCallback, useEffect, useState } from 'react';
import { OverviewPanel } from './components/graph/GraphLegend';
import { KGPanel } from './components/kg/KGPanel';
import { ModelSelector } from './components/kg/ModelSelector';
import { Neo4jForm } from './components/kg/Neo4jForm';
import { MainLayout } from './components/layout/MainLayout';
import { Sidebar } from './components/layout/Sidebar';
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
      if (result.kg_name) localStorage.setItem('currentKGName', result.kg_name);
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
    <div className="flex h-screen w-screen overflow-hidden bg-base-100 text-base-content">
      {state.sidebarCollapsed && (
        <button
          type="button"
          className="absolute top-2 left-1 z-30 btn btn-ghost btn-xs"
          onClick={() => dispatch({ type: 'TOGGLE_SIDEBAR' })}
          title="Expand sidebar"
        >
          ›
        </button>
      )}

      <Sidebar>
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

      <MainLayout
        layout={state.layout ?? (state.kgExpanded ? 'graph-only' : 'split')}
        graphPanel={
          <Suspense fallback={<PanelSkeleton />}>
            <GraphContainer progressActive={progressActive} onProgressClose={() => setProgressActive(false)} />
          </Suspense>
        }
        chatPanel={
          <Suspense fallback={<PanelSkeleton />}>
            <ChatPanel ragModelHook={ragModelHook} />
          </Suspense>
        }
      />

      <Neo4jForm open={neo4jOpen} onClose={() => setNeo4jOpen(false)} onLoaded={handleNeo4jLoaded} />
      <Notifications />
    </div>
  );
}
