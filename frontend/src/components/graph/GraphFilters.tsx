import { useEffect, useMemo, useState } from 'react';
import { useApp } from '../../context/AppContext';

export function GraphFilters() {
  const { state, dispatch } = useApp();
  const [open, setOpen] = useState(false);
  const [checkedNodes, setCheckedNodes] = useState<Set<string>>(new Set());
  const [checkedRels, setCheckedRels] = useState<Set<string>>(new Set());

  const nodeTypes = useMemo(() => Object.keys(state.nodeTypeColors), [state.nodeTypeColors]);
  const relTypes = useMemo(() => Object.keys(state.relationshipTypeColors), [state.relationshipTypeColors]);

  useEffect(() => {
    if (nodeTypes.length > 0) {
      setCheckedNodes(new Set(nodeTypes));
    }
    if (relTypes.length > 0) {
      setCheckedRels(new Set(relTypes));
    }
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
    setOpen(false);
  };

  const resetFilters = () => {
    setCheckedNodes(new Set(nodeTypes));
    setCheckedRels(new Set(relTypes));
    dispatch({ type: 'CLEAR_FILTERS' });
    setOpen(false);
  };

  return (
    <>
      <button type="button" className="btn btn-ghost btn-sm" onClick={() => setOpen(!open)}>
        Filters
      </button>

      {open && (
        <div className="absolute top-full right-0 mt-1 bg-base-100 border border-base-300 rounded-lg shadow-lg z-30 p-3 w-64 max-h-80 overflow-y-auto">
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
      )}
    </>
  );
}
