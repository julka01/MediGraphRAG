interface NodeDetailPanelProps {
  node: Record<string, unknown>;
  nodeColor: string;
  onClose: () => void;
}

export function NodeDetailPanel({ node, nodeColor, onClose }: NodeDetailPanelProps) {
  if (!node) return null;

  const labels = node.labels as string[] | undefined;
  const nodeType = labels?.[0] || 'Unknown';
  const props = (node.properties || {}) as Record<string, unknown>;

  return (
    <div className="absolute top-0 right-0 w-72 h-full bg-base-100 border-l border-base-300 shadow-lg z-20 overflow-y-auto p-4">
      <button className="btn btn-ghost btn-xs btn-circle absolute top-2 right-2" onClick={onClose}>✕</button>

      <div className="font-bold text-sm mt-4 mb-1">{(node.label as string) || String(node.originalId)}</div>

      <span className="badge badge-sm" style={{ background: nodeColor + '33', color: nodeColor, border: `1px solid ${nodeColor}55` }}>
        {nodeType}
      </span>

      <div className="font-semibold text-xs mt-4 mb-2 opacity-70">Properties</div>
      {Object.keys(props).length > 0 ? (
        <div className="space-y-1">
          {Object.entries(props).map(([k, v]) => (
            <div key={k} className="flex gap-2 text-xs">
              <span className="font-medium opacity-70 shrink-0">{k}</span>
              <span className="break-all">{String(v)}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-xs opacity-50">No additional properties</div>
      )}
    </div>
  );
}
