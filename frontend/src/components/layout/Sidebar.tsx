import { useApp } from '../../context/AppContext';
import { useTheme } from '../../context/ThemeContext';
import type { HealthResponse, UseModelsReturn } from '../../types/app';
import { OverviewPanel } from '../graph/GraphLegend';
import { KGPanel } from '../kg/KGPanel';
import { ModelSelector } from '../kg/ModelSelector';
import { HealthDot } from '../ui/HealthDot';

interface SidebarProps {
  kgModelHook: UseModelsReturn;
  ragModelHook: UseModelsReturn;
  onNeo4jOpen: () => void;
  onProgressStart: () => void;
  onProgressStop: () => void;
  healthData?: HealthResponse | null;
}

export function Sidebar({
  kgModelHook,
  ragModelHook,
  onNeo4jOpen,
  onProgressStart,
  onProgressStop,
  healthData,
}: SidebarProps) {
  const { state, dispatch } = useApp();
  const { theme, toggleTheme } = useTheme();

  return (
    <div
      className={`relative flex flex-col bg-base-200 border-r border-base-300 transition-all duration-300 overflow-y-auto ${
        state.sidebarCollapsed ? 'w-0 min-w-0 overflow-hidden' : 'w-72 min-w-72'
      }`}
    >
      <button
        type="button"
        className="absolute top-2 -right-6 z-30 btn btn-ghost btn-xs"
        onClick={() => dispatch({ type: 'TOGGLE_SIDEBAR' })}
        title={state.sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {state.sidebarCollapsed ? '›' : '‹'}
      </button>

      <div className="p-3 space-y-4">
        <div className="flex items-center justify-between">
          <button type="button" className="btn btn-ghost btn-sm" onClick={toggleTheme}>
            {theme === 'dark' ? '🌙' : '☀️'}
          </button>
          <HealthDot initialData={healthData} />
        </div>

        <div>
          <h3 className="font-semibold text-sm mb-2">Knowledge Graph</h3>
          <ModelSelector label="KG" vendorHook={kgModelHook} />
          <div className="mt-2">
            <KGPanel
              kgModelHook={kgModelHook}
              onNeo4jOpen={onNeo4jOpen}
              onProgressStart={onProgressStart}
              onProgressStop={onProgressStop}
            />
          </div>
        </div>

        <div>
          <h3 className="font-semibold text-sm mb-2">RAG Model</h3>
          <ModelSelector label="RAG" vendorHook={ragModelHook} />
        </div>

        <OverviewPanel />
      </div>
    </div>
  );
}
