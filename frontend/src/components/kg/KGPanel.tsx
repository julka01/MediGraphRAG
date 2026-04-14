// frontend/src/components/kg/KGPanel.tsx
import type { KGSettings, UseModelsReturn } from '../../types/app';
import { KGBuildSection } from './KGBuildSection';
import { KGLoadSection } from './KGLoadSection';

interface KGPanelProps {
  kgModelHook: UseModelsReturn;
  onLoadKG: (loadMode: string, nodeLimit: number, kgFilter: string) => void;
  onProgressStart: () => void;
  onProgressStop: () => void;
  loadedKGSettings?: KGSettings | null;
}

export function KGPanel({ kgModelHook, onLoadKG, onProgressStart, onProgressStop, loadedKGSettings }: KGPanelProps) {
  return (
    <div className="flex flex-col gap-4 min-w-0">
      <div className="px-1">
        <p className="text-[0.65rem] font-medium uppercase tracking-[0.22em] text-primary/70">Workspace</p>
        <h1 className="mt-2 text-lg font-semibold text-base-content">OntoGraphRAG</h1>
      </div>

      <section className="panel-glass rounded-3xl p-4">
        <div className="mb-4">
          <p className="text-[0.65rem] font-medium uppercase tracking-[0.18em] text-base-content/45">Build</p>
          <h2 className="mt-1 text-sm font-semibold text-base-content">Create a new graph</h2>
          <p className="mt-1 text-xs leading-5 text-base-content/58">
            Upload a source document and ontology, choose the extraction model, and generate a new KG.
          </p>
        </div>
        <KGBuildSection
          kgModelHook={kgModelHook}
          onProgressStart={onProgressStart}
          onProgressStop={onProgressStop}
          loadedKGSettings={loadedKGSettings}
        />
      </section>

      <section className="panel-glass rounded-3xl p-4">
        <div className="mb-4">
          <p className="text-[0.65rem] font-medium uppercase tracking-[0.18em] text-base-content/45">Explore</p>
          <h2 className="mt-1 text-sm font-semibold text-base-content">Load an existing graph</h2>
          <p className="mt-1 text-xs leading-5 text-base-content/58">
            Reopen a saved KG, sample a lighter view for large graphs, or remove stale graphs.
          </p>
        </div>
        <KGLoadSection onLoadKG={onLoadKG} />
      </section>
    </div>
  );
}
