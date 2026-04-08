import { useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import type { GraphNode, GraphRelationship } from '../../types/app';

export function OverviewPanel() {
  const { state, dispatch } = useApp();

  const { nodeCounts, relCounts } = useMemo(() => {
    const nc: Record<string, number> = {};
    const rc: Record<string, number> = {};
    (state.graphData?.nodes || []).forEach((n: GraphNode) => {
      const label = n.labels?.[0] || 'Unknown';
      nc[label] = (nc[label] || 0) + 1;
    });
    (state.graphData?.relationships || []).forEach((r: GraphRelationship) => {
      const type = r.type || 'Unknown';
      rc[type] = (rc[type] || 0) + 1;
    });
    return { nodeCounts: nc, relCounts: rc };
  }, [state.graphData]);

  if (!state.graphData) return null;

  const handleNodeFilter = (label: string) => {
    dispatch({ type: 'SET_FILTERS', nodeTypes: [label], relationshipTypes: [] });
  };

  const handleRelFilter = (type: string) => {
    dispatch({ type: 'SET_FILTERS', nodeTypes: [], relationshipTypes: [type] });
  };

  return (
    <div className="mt-2">
      <div className="collapse collapse-arrow">
        <input type="checkbox" defaultChecked />
        <div className="collapse-title text-sm font-semibold py-1 min-h-0">Details</div>
        <div className="collapse-content">
          <div className="space-y-2 text-xs">
            <div>
              <div className="font-semibold mb-1 opacity-70">Node Labels</div>
              {Object.entries(nodeCounts)
                .sort(([, a], [, b]) => b - a)
                .map(([label, count]) => (
                  <button
                    type="button"
                    key={label}
                    className="flex items-center justify-between py-0.5 cursor-pointer hover:bg-base-200 rounded px-1 w-full text-left bg-transparent border-0"
                    onClick={() => handleNodeFilter(label)}
                  >
                    <span className="flex items-center gap-1.5">
                      <span
                        className="size-2.5 rounded-full"
                        style={{ backgroundColor: state.nodeTypeColors[label] || 'var(--color-primary)' }}
                      />
                      {label}
                    </span>
                    <span className="badge badge-xs">{count}</span>
                  </button>
                ))}
            </div>
            <div>
              <div className="font-semibold mb-1 opacity-70">Relationship Types</div>
              {Object.entries(relCounts)
                .sort(([, a], [, b]) => b - a)
                .map(([type, count]) => (
                  <button
                    type="button"
                    key={type}
                    className="flex items-center justify-between py-0.5 cursor-pointer hover:bg-base-200 rounded px-1 w-full text-left bg-transparent border-0"
                    onClick={() => handleRelFilter(type)}
                  >
                    <span className="flex items-center gap-1.5">
                      <span
                        className="size-2.5 rounded-full"
                        style={{ backgroundColor: state.relationshipTypeColors[type] || 'var(--color-neutral)' }}
                      />
                      {type}
                    </span>
                    <span className="badge badge-xs">{count}</span>
                  </button>
                ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
