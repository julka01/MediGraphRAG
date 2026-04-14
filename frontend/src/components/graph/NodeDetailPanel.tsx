import { useState } from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';

interface NodeEdge {
  from: string;
  to: string;
  label: string;
  toLabel: string;
  fromLabel: string;
}

interface NodeDetailPanelProps {
  node: Record<string, unknown>;
  nodeColor: string;
  edges: NodeEdge[];
  onClose: () => void;
}

export function NodeDetailPanel({ node, nodeColor, edges, onClose }: NodeDetailPanelProps) {
  if (!node) return null;

  const labels = node.labels as string[] | undefined;
  const nodeType = labels?.[0] || 'Unknown';
  const nodeName = (node.label as string) || String(node.originalId);
  const props = (node.properties || {}) as Record<string, unknown>;

  const [showAllConnections, setShowAllConnections] = useState(false);
  const currentNodeId = String(node.id ?? node.originalId);
  const totalConnections = edges.length;
  const outEdges = edges.filter((e) => e.from === currentNodeId);
  const inEdges = edges.filter((e) => e.to === currentNodeId);
  const outCount = outEdges.length;
  const inCount = inEdges.length;

  return (
    <div className="absolute top-0 right-0 w-72 h-full z-20 motion-safe:animate-slide-from-right flex">
      <div className="w-1 shrink-0 bg-base-300" />
      <div className="card bg-base-100 h-full shadow-lg rounded-none overflow-hidden flex flex-col flex-1">
        {/* Header */}
        <div className="px-4 py-3 bg-base-200 shrink-0">
          <div className="flex justify-between items-start">
            <div className="pr-6">
              <div className="font-bold text-base leading-tight">{nodeName}</div>
              <span className="badge badge-sm bg-base-300 text-base-content gap-1 mt-1 text-2xs">
                <span className="size-2 rounded-full shrink-0" style={{ backgroundColor: nodeColor }} />
                {nodeType}
              </span>
            </div>
            <button
              type="button"
              className="btn btn-ghost btn-xs btn-circle shrink-0"
              onClick={onClose}
              aria-label="Close node details"
            >
              <XMarkIcon className="size-4" aria-hidden="true" />
            </button>
          </div>
        </div>

        {/* Quick stats row */}
        <div className="flex border-b border-base-300 text-center text-2xs text-base-content/60 shrink-0 tabular-nums">
          <div className="flex-1 py-2 border-r border-base-300">
            <div className="text-base font-semibold text-base-content">{totalConnections}</div>
            connections
          </div>
          <div className="flex-1 py-2 border-r border-base-300">
            <div className="text-base font-semibold text-base-content">{inCount}</div>
            in
          </div>
          <div className="flex-1 py-2">
            <div className="text-base font-semibold text-base-content">{outCount}</div>
            out
          </div>
        </div>

        {/* Scrollable body */}
        <div className="overflow-y-auto flex-1">
          {/* Properties table */}
          <div className="px-4 py-3">
            <div className="text-2xs uppercase tracking-wider text-base-content/40 mb-2">Properties</div>
            {Object.keys(props).length > 0 ? (
              <table className="table table-xs w-full">
                <tbody>
                  {Object.entries(props).map(([k, v]) => (
                    <tr key={k} className="border-b border-base-300/30">
                      <td className="py-1.5 pr-3 text-base-content/50 whitespace-nowrap align-top w-24">{k}</td>
                      <td className="py-1.5 break-all">
                        {k === 'confidence' && typeof v === 'number' ? (
                          <div className="flex items-center gap-2">
                            <div className="flex-1 h-1 bg-base-300 rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full"
                                style={{ width: `${v * 100}%`, backgroundColor: nodeColor }}
                              />
                            </div>
                            <span className="text-2xs text-base-content/50">{v.toFixed(2)}</span>
                          </div>
                        ) : (
                          <span
                            className={
                              /^[a-z0-9_-]+$/i.test(String(v)) && String(v).length > 8 ? 'font-mono text-2xs' : ''
                            }
                          >
                            {String(v)}
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="text-xs opacity-50">No additional properties</div>
            )}
          </div>

          {/* Connections preview */}
          {edges.length > 0 && (
            <div className="px-4 pb-3">
              <div className="text-2xs uppercase tracking-wider text-base-content/40 mb-2">Connections</div>
              <div className="space-y-1">
                {(showAllConnections ? edges : edges.slice(0, 5)).map((edge, i) => (
                  <div key={i} className="flex items-center gap-1.5 rounded bg-base-300/30 px-2 py-1 text-xs">
                    <span
                      className={`badge badge-xs shrink-0 ${edge.from === currentNodeId ? 'badge-primary badge-outline' : 'badge-ghost'}`}
                    >
                      {edge.from === currentNodeId ? 'OUT' : 'IN'}
                    </span>
                    <span className="text-base-content/50 shrink-0">
                      {edge.from === currentNodeId ? `${edge.label} →` : `← ${edge.label}`}
                    </span>
                    <span className="truncate">
                      {edge.from === currentNodeId ? edge.toLabel || edge.to : edge.fromLabel || edge.from}
                    </span>
                  </div>
                ))}
                {edges.length > 5 && (
                  <button
                    type="button"
                    className="text-2xs text-base-content/40 hover:text-base-content/60 transition-colors w-full text-right mt-1"
                    onClick={() => setShowAllConnections(!showAllConnections)}
                  >
                    {showAllConnections ? 'Show less' : `Show ${edges.length - 5} more`}
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
