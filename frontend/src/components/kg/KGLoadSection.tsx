// frontend/src/components/kg/KGLoadSection.tsx
import clsx from 'clsx';
import { useCallback, useState } from 'react';
import { api } from '../../api';
import { useApp } from '../../context/AppContext';
import { FieldsetDropdown } from '../ui/FieldsetDropdown';
import { showError, showSuccess } from '../ui/Notifications';

interface KGLoadSectionProps {
  onLoadKG: (loadMode: string, nodeLimit: number, kgFilter: string) => void;
}

export function KGLoadSection({ onLoadKG }: KGLoadSectionProps) {
  const { state, dispatch } = useApp();
  const [importMode, setImportMode] = useState('limited-1000');
  const [kgFilter, setKgFilter] = useState('');

  const handleLoadKG = () => {
    let loadMode: string;
    let nodeLimit: number;
    if (importMode === 'sample') {
      loadMode = 'sample';
      nodeLimit = 0;
    } else if (importMode === 'complete') {
      loadMode = 'complete';
      nodeLimit = 0;
    } else {
      loadMode = 'limited';
      nodeLimit = Number(importMode.split('-')[1]);
    }
    onLoadKG(loadMode, nodeLimit, kgFilter);
  };

  const handleDelete = async () => {
    const targetName = kgFilter || state.currentKGName;
    const label = targetName || 'all knowledge graphs';
    if (!window.confirm(`Delete "${label}"? This cannot be undone.`)) return;
    try {
      const result = await api.clearKG(targetName || undefined);
      const isLoadedKG = !targetName || targetName === state.currentKGName;
      if (isLoadedKG) {
        dispatch({ type: 'CLEAR_KG' });
        localStorage.removeItem('currentKGName');
      }
      if (targetName) {
        dispatch({ type: 'SET_KG_LIST', kgList: state.kgList.filter((kg) => kg.name !== targetName) });
      } else {
        dispatch({ type: 'SET_KG_LIST', kgList: [] });
      }
      setKgFilter('');
      showSuccess(dispatch, result.message || 'KG cleared');
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      showError(dispatch, `Failed to clear KG: ${msg}`);
    }
  };

  const refreshKGList = useCallback(() => {
    api.fetchKGList().then(
      (res) => dispatch({ type: 'SET_KG_LIST', kgList: res.kgs || [] }),
      () => {},
    );
  }, [dispatch]);

  return (
    <>
      {/* Import Mode */}
      <FieldsetDropdown
        label="Import Mode"
        options={[
          { value: 'limited-500', label: 'First 500 nodes' },
          { value: 'limited-1000', label: 'First 1000 nodes' },
          { value: 'sample', label: 'Top hubs (smart sample)' },
          { value: 'complete', label: 'Complete graph' },
        ]}
        value={importMode}
        onChange={setImportMode}
      />

      {/* KG Name Filter */}
      <FieldsetDropdown
        label="KG Name"
        placeholder="Select KG"
        options={state.kgList.map((kg) => ({ value: kg.name, label: kg.name }))}
        value={kgFilter}
        onChange={setKgFilter}
        onOpen={refreshKGList}
      />

      {/* Load KG + Delete KG */}
      <div className="flex gap-2">
        <button
          type="button"
          className={clsx(
            'btn btn-sm flex-1 shadow-none',
            kgFilter
              ? 'bg-transparent border border-primary/50 text-primary hover:bg-primary hover:text-primary-content'
              : 'bg-transparent border border-base-content/20 text-base-content/30 pointer-events-none',
          )}
          onClick={handleLoadKG}
          disabled={!kgFilter}
        >
          Load KG
        </button>
        <button
          type="button"
          className={clsx(
            'btn btn-sm flex-1',
            kgFilter || state.currentKGId
              ? 'btn-outline border-error/50 text-error hover:bg-error hover:text-error-content'
              : 'bg-transparent border border-base-content/20 text-base-content/30 pointer-events-none',
          )}
          onClick={handleDelete}
          disabled={!kgFilter && !state.currentKGId}
        >
          Delete KG
        </button>
      </div>
    </>
  );
}
