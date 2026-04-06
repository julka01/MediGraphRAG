import { useState, useMemo } from 'react';
import { useApp } from '../../context/AppContext';

export function GraphLegend() {
  const { state } = useApp();
  const [collapsed, setCollapsed] = useState(false);

  const nodeEntries = Object.entries(state.nodeTypeColors);
  const relEntries = Object.entries(state.relationshipTypeColors);

  if (nodeEntries.length === 0 && relEntries.length === 0) return null;

  return (
    <div className="absolute bottom-2 left-2 bg-base-100/90 backdrop-blur border border-base-300 rounded-lg p-2 z-10 max-w-48 text-xs">
      <div className="font-semibold mb-1 cursor-pointer flex items-center justify-between" onClick={() => setCollapsed(!collapsed)}>
        <span>Graph Legend</span>
        <span>{collapsed ? '▶' : '▼'}</span>
      </div>
      {!collapsed && (
        <div className="space-y-0.5 max-h-48 overflow-y-auto">
          {nodeEntries.map(([type, color]) => (
            <div key={type} className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: color }} />
              <span className="truncate">{type}</span>
            </div>
          ))}
          {relEntries.length > 0 && nodeEntries.length > 0 && (
            <hr className="my-1 border-base-300" />
          )}
          {relEntries.map(([type, color]) => (
            <div key={type} className="flex items-center gap-1.5">
              <span className="w-3 h-1 shrink-0 rounded" style={{ backgroundColor: color }} />
              <span className="truncate">{type} →</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function MiniLegend() {
  const { state } = useApp();
  const entries = Object.entries(state.nodeTypeColors).slice(0, 14);

  if (entries.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] opacity-80 mb-1">
      {entries.map(([type, color]) => (
        <span key={type} className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full inline-block" style={{ background: color }} />
          {type}
        </span>
      ))}
    </div>
  );
}

export function OverviewPanel() {
  const { state, dispatch } = useApp();
  const [collapsed, setCollapsed] = useState(false);

  const { nodeCounts, relCounts } = useMemo(() => {
    const nc = {};
    const rc = {};
    (state.graphData?.nodes || []).forEach((n) => {
      const label = n.labels?.[0] || 'Unknown';
      nc[label] = (nc[label] || 0) + 1;
    });
    (state.graphData?.relationships || []).forEach((r) => {
      const type = r.type || 'Unknown';
      rc[type] = (rc[type] || 0) + 1;
    });
    return { nodeCounts: nc, relCounts: rc };
  }, [state.graphData]);

  if (!state.graphData) return null;

  const handleNodeFilter = (label) => {
    dispatch({ type: 'SET_FILTERS', nodeTypes: [label], relationshipTypes: [] });
  };

  const handleRelFilter = (type) => {
    dispatch({ type: 'SET_FILTERS', nodeTypes: [], relationshipTypes: [type] });
  };

  return (
    <div className="mt-2">
      <div className="flex items-center justify-between cursor-pointer text-sm font-semibold py-1" onClick={() => setCollapsed(!collapsed)}>
        <span>Details</span>
        <span className="text-xs">{collapsed ? '▶' : '▼'}</span>
      </div>
      {!collapsed && (
        <div className="space-y-2 text-xs">
          <div>
            <div className="font-semibold mb-1 opacity-70">Node Labels</div>
            {Object.entries(nodeCounts)
              .sort(([, a], [, b]) => b - a)
              .map(([label, count]) => (
                <div key={label} className="flex items-center justify-between py-0.5 cursor-pointer hover:bg-base-200 rounded px-1" onClick={() => handleNodeFilter(label)}>
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: state.nodeTypeColors[label] || '#428bca' }} />
                    {label}
                  </span>
                  <span className="badge badge-xs">{count}</span>
                </div>
              ))}
          </div>
          <div>
            <div className="font-semibold mb-1 opacity-70">Relationship Types</div>
            {Object.entries(relCounts)
              .sort(([, a], [, b]) => b - a)
              .map(([type, count]) => (
                <div key={type} className="flex items-center justify-between py-0.5 cursor-pointer hover:bg-base-200 rounded px-1" onClick={() => handleRelFilter(type)}>
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: state.relationshipTypeColors[type] || '#888' }} />
                    {type}
                  </span>
                  <span className="badge badge-xs">{count}</span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
