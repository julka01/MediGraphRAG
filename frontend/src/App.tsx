import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { BottomBar } from './components/graph/BottomBar';
import { TopBar } from './components/graph/TopBar';
import { KGPanel } from './components/kg/KGPanel';
import { MainLayout } from './components/layout/MainLayout';
import { Sidebar } from './components/layout/Sidebar';
import { ErrorBoundary } from './components/ui/ErrorBoundary';
import { Notifications } from './components/ui/Notifications';
import { useApp } from './context/AppContext';
import { useTheme } from './context/ThemeContext';
import { useKGLoader } from './hooks/useKGLoader';
import { useModels } from './hooks/useModels';
import { useSnapToClose } from './hooks/useSnapToClose';
import { useStartup } from './hooks/useStartup';

const GraphContainer = lazy(() =>
  import('./components/graph/GraphContainer').then((m) => ({ default: m.GraphContainer })),
);
const ChatPanel = lazy(() => import('./components/chat/ChatPanel').then((m) => ({ default: m.ChatPanel })));

function PanelSkeleton() {
  return (
    <div className="flex flex-col h-full p-4 gap-4">
      <div className="skeleton h-6 w-40" />
      <div className="skeleton flex-1" />
    </div>
  );
}

export default function App() {
  const { state, dispatch } = useApp();
  const { theme } = useTheme();

  const startup = useStartup(['openai', 'openrouter']);
  const kgModelHook = useModels('openai', startup.modelsByVendor.openai);
  const ragModelHook = useModels('openai', startup.modelsByVendor.openai);

  const { loadedKGSettings, handleLoadKG } = useKGLoader(kgModelHook);

  const [progressActive, setProgressActive] = useState(false);

  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        const textarea = document.querySelector<HTMLTextAreaElement>('textarea[name="chat-question"]');
        if (textarea) {
          textarea.focus();
          textarea.select();
        }
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

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

      {/* Left sidebar — always mounted to preserve form state */}
      <Sidebar hidden={state.panels.leftCollapsed}>
        <KGPanel
          kgModelHook={kgModelHook}
          onLoadKG={handleLoadKG}
          onProgressStart={() => setProgressActive(true)}
          onProgressStop={() => setProgressActive(false)}
          loadedKGSettings={loadedKGSettings}
        />
      </Sidebar>
      {!state.panels.leftCollapsed && (
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize left panel"
          className="app-divider-v hidden md:block w-1 shrink-0 cursor-col-resize transition-colors"
          onPointerDown={leftSnap.onPointerDown}
        />
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
          topBar={<TopBar />}
        />
      </div>

      <Notifications />
    </div>
  );
}
