import { useCallback, useEffect, useState } from 'react';
import { api } from './api';
import { ChatPanel } from './components/chat/ChatPanel';
import { GraphContainer } from './components/graph/GraphContainer';
import { Neo4jForm } from './components/kg/Neo4jForm';
import { Sidebar } from './components/layout/Sidebar';
import { Notifications, showSuccess } from './components/ui/Notifications';
import { useApp } from './context/AppContext';
import { useTheme } from './context/ThemeContext';
import { useModels } from './hooks/useModels';
import type { LoadNeo4jResponse, Neo4jStats } from './types/app';

export default function App() {
  const { state, dispatch } = useApp();
  const { theme } = useTheme();
  const kgModelHook = useModels('openai');
  const ragModelHook = useModels('openrouter');

  const [neo4jOpen, setNeo4jOpen] = useState(false);
  const [progressActive, setProgressActive] = useState(false);

  useEffect(() => {
    api
      .fetchKGList()
      .then((data) => dispatch({ type: 'SET_KG_LIST', kgList: data.kgs || [] }))
      .catch(() => {});
  }, [dispatch]);

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

  const chatHidden = state.kgExpanded;

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

      <Sidebar
        kgModelHook={kgModelHook}
        ragModelHook={ragModelHook}
        onNeo4jOpen={() => setNeo4jOpen(true)}
        onProgressStart={() => setProgressActive(true)}
        onProgressStop={() => setProgressActive(false)}
      />

      <div className="flex flex-1 min-w-0">
        <div className={`border-r border-base-300 overflow-hidden ${chatHidden ? 'flex-1' : 'flex-[1.6]'}`}>
          <GraphContainer progressActive={progressActive} onProgressClose={() => setProgressActive(false)} />
        </div>
        {!chatHidden && (
          <div className="overflow-hidden flex-1">
            <ChatPanel ragModelHook={ragModelHook} />
          </div>
        )}
      </div>

      <Neo4jForm open={neo4jOpen} onClose={() => setNeo4jOpen(false)} onLoaded={handleNeo4jLoaded} />
      <Notifications />
    </div>
  );
}
