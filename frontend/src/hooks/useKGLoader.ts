import { useCallback, useState } from 'react';
import { api } from '../api';
import { showError, showSuccess } from '../components/ui/Notifications';
import { useApp } from '../context/AppContext';
import type { KGSettings, LoadNeo4jResponse, Neo4jStats, UseModelsReturn } from '../types/app';
import { safeSet } from '../utils/storage';

export interface UseKGLoaderReturn {
  loadedKGSettings: KGSettings | null;
  handleLoadKG: (loadMode: string, nodeLimit: number, kgFilter: string) => Promise<void>;
}

export function useKGLoader(kgModelHook: UseModelsReturn): UseKGLoaderReturn {
  const { state, dispatch } = useApp();
  const [loadedKGSettings, setLoadedKGSettings] = useState<KGSettings | null>(null);
  const { restoreVendorModel } = kgModelHook;

  const handleNeo4jLoaded = useCallback(
    (result: LoadNeo4jResponse, kgFilter: string, stats: Neo4jStats) => {
      dispatch({ type: 'SET_KG', kgId: result.kg_id, kgName: result.kg_name || kgFilter || null });
      if (result.kg_name) safeSet('currentKGName', result.kg_name);
      dispatch({ type: 'SET_GRAPH_DATA', data: result.graph_data });
      dispatch({ type: 'CLEAR_HIGHLIGHTED_NODES' });
      dispatch({ type: 'CLEAR_FILTERS' });

      // Restore KG creation settings
      const settings = result.kg_settings;
      if (settings) {
        if (settings.provider && settings.model) {
          restoreVendorModel(settings.provider, settings.model);
        }
        setLoadedKGSettings(settings);
      }

      const nodeCount = result.graph_data?.nodes?.length ?? 0;
      const relCount = result.graph_data?.relationships?.length ?? 0;
      let msg = result.message || `Loaded KG from Neo4j with ${nodeCount} nodes and ${relCount} relationships`;
      if (stats?.sample_mode) msg += ' (Smart Sample)';
      else if (stats?.complete_import) msg += ' (Complete Import)';
      showSuccess(dispatch, msg);
    },
    [dispatch, restoreVendorModel],
  );

  const handleLoadKG = useCallback(
    async (loadMode: string, nodeLimit: number, kgFilter: string) => {
      const formData = new FormData();
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
        if (kgFilter && !result.graph_data?.nodes?.length) {
          showError(dispatch, `KG "${kgFilter}" not found`);
          dispatch({ type: 'SET_KG_LIST', kgList: state.kgList.filter((kg) => kg.name !== kgFilter) });
          return;
        }
        handleNeo4jLoaded(result, kgFilter, result.stats ?? ({} as Neo4jStats));
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        showError(dispatch, `Loading failed: ${msg}`);
      }
    },
    [dispatch, handleNeo4jLoaded, state.kgList],
  );

  return { loadedKGSettings, handleLoadKG };
}
