import { useState, useCallback } from 'react';
import { useApp } from '../../context/AppContext';

export function GraphSearch() {
  const { networkRef } = useApp();
  const [term, setTerm] = useState('');
  const [matchCount, setMatchCount] = useState(null);

  const performSearch = useCallback((searchTerm) => {
    const network = networkRef.current;
    if (!network) return;

    const nodeDS = network.body.data.nodes;
    const edgeDS = network.body.data.edges;
    const t = (searchTerm || '').toLowerCase().trim();

    if (!t) {
      nodeDS.update(nodeDS.get().map((n) => ({ id: n.id, opacity: 1, hidden: false })));
      edgeDS.update(edgeDS.get().map((e) => ({ id: e.id, opacity: 1, hidden: false })));
      setMatchCount(null);
      network.redraw();
      return;
    }

    const matched = new Set();
    const nodeUpdates = nodeDS.get().map((node) => {
      const hit =
        (node.label || '').toLowerCase().includes(t) ||
        (node.title || '').toLowerCase().includes(t) ||
        JSON.stringify(node.properties || {}).toLowerCase().includes(t);
      if (hit) matched.add(node.id);
      return { id: node.id, hidden: false, opacity: hit ? 1 : 0.1 };
    });
    nodeDS.update(nodeUpdates);

    const edgeUpdates = edgeDS.get().map((edge) => ({
      id: edge.id,
      hidden: false,
      opacity: matched.has(edge.from) || matched.has(edge.to) ? 1 : 0.06,
    }));
    edgeDS.update(edgeUpdates);

    setMatchCount(matched.size);
    network.redraw();
  }, [networkRef]);

  const handleInput = (e) => {
    const value = e.target.value;
    setTerm(value);
    performSearch(value);
  };

  const handleClear = () => {
    setTerm('');
    performSearch('');
  };

  return (
    <div className="relative">
      <input
        type="text"
        className="input input-bordered input-sm w-44 pr-14"
        placeholder="Search nodes…"
        value={term}
        onChange={handleInput}
        autoComplete="off"
      />
      {term && (
        <button className="absolute right-8 top-1/2 -translate-y-1/2 btn btn-ghost btn-xs px-1" onClick={handleClear}>×</button>
      )}
      {matchCount !== null && matchCount > 0 && (
        <span className="absolute right-1 top-1/2 -translate-y-1/2 text-[10px] font-semibold text-accent pointer-events-none">{matchCount}</span>
      )}
    </div>
  );
}
