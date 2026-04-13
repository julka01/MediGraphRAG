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
    <div className="flex flex-col gap-3 min-w-0">
      {/* ── Build Section ─────────────────────────────────── */}
      <KGBuildSection
        kgModelHook={kgModelHook}
        onProgressStart={onProgressStart}
        onProgressStop={onProgressStop}
        loadedKGSettings={loadedKGSettings}
      />

      <div className="flex items-center gap-3 my-2">
        <div className="h-px flex-1 bg-gradient-to-r from-transparent to-base-content/20" />
        <div className="size-1 rounded-full bg-base-content/20" />
        <div className="h-px flex-1 bg-gradient-to-l from-transparent to-base-content/20" />
      </div>

      {/* ── Load Section ──────────────────────────────────── */}
      <KGLoadSection onLoadKG={onLoadKG} />
    </div>
  );
}
