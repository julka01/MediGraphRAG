import { FunnelIcon } from '@heroicons/react/24/outline';
import { useEffect, useMemo, useState } from 'react';
import { useApp } from '../../context/AppContext';

export function GraphFilters() {
  const { state, dispatch } = useApp();
  const [checkedNodes, setCheckedNodes] = useState<Set<string>>(new Set());
  const [checkedRels, setCheckedRels] = useState<Set<string>>(new Set());

  const nodeTypes = useMemo(() => Object.keys(state.nodeTypeColors), [state.nodeTypeColors]);
  const relTypes = useMemo(() => Object.keys(state.relationshipTypeColors), [state.relationshipTypeColors]);

  useEffect(() => {
    if (nodeTypes.length > 0) setCheckedNodes(new Set(nodeTypes));
    if (relTypes.length > 0) setCheckedRels(new Set(relTypes));
  }, [nodeTypes, relTypes]);

  const toggleNode = (type: string) => {
    setCheckedNodes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  const toggleRel = (type: string) => {
    setCheckedRels((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  const applyFilters = () => {
    dispatch({ type: 'SET_FILTERS', nodeTypes: [...checkedNodes], relationshipTypes: [...checkedRels] });
    (document.activeElement as HTMLElement)?.blur();
  };

  const resetFilters = () => {
    setCheckedNodes(new Set(nodeTypes));
    setCheckedRels(new Set(relTypes));
    dispatch({ type: 'CLEAR_FILTERS' });
    (document.activeElement as HTMLElement)?.blur();
  };

  const activeFilterCount =
    (nodeTypes.length - checkedNodes.size) + (relTypes.length - checkedRels.size);

  return (
    <div className="dropdown dropdown-end">
      <button type="button" className="btn btn-ghost btn-sm" tabIndex={0}>
        <FunnelIcon className="size-4" aria-hidden="true" />
        Filters
        {activeFilterCount > 0 && (
          <span className="badge badge-xs badge-primary">{activeFilterCount}</span>
        )}
      </button>
      <div className="dropdown-content card card-border bg-base-100 shadow-lg z-30 w-64" tabIndex={0}>
        <div className="card-body p-3 max-h-80 overflow-y-auto">
          <h4 className="font-semibold text-sm mb-2">Graph Filters</h4>

          {nodeTypes.length > 0 && (
            <div className="mb-3">
              <div className="text-xs font-semibold mb-1 opacity-70">Node Types</div>
              {nodeTypes.map((type) => (
                <label key={type} className="flex items-center gap-2 cursor-pointer py-0.5">
                  <input
                    type="checkbox"
                    className="checkbox checkbox-xs"
                    checked={checkedNodes.has(type)}
                    onChange={() => toggleNode(type)}
                  />
                  <span
                    className="size-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: state.nodeTypeColors[type] }}
                  />
                  <span className="text-xs">{type}</span>
                </label>
              ))}
            </div>
          )}

          {relTypes.length > 0 && (
            <div className="mb-3">
              <div className="text-xs font-semibold mb-1 opacity-70">Relationship Types</div>
              {relTypes.map((type) => (
                <label key={type} className="flex items-center gap-2 cursor-pointer py-0.5">
                  <input
                    type="checkbox"
                    className="checkbox checkbox-xs"
                    checked={checkedRels.has(type)}
                    onChange={() => toggleRel(type)}
                  />
                  <span className="text-xs">{type}</span>
                </label>
              ))}
            </div>
          )}

          <div className="flex gap-2">
            <button type="button" className="btn btn-primary btn-xs flex-1" onClick={applyFilters}>
              Apply
            </button>
            <button type="button" className="btn btn-ghost btn-xs flex-1" onClick={resetFilters}>
              Reset
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
