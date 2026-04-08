import { memo, useMemo, useRef, useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';

export const FilterChips = memo(function FilterChips() {
  const { state, dispatch } = useApp();
  const containerRef = useRef<HTMLDivElement>(null);
  const [visibleCount, setVisibleCount] = useState<number | null>(null);

  const nodeTypes = useMemo(
    () => Object.entries(state.nodeTypeColors),
    [state.nodeTypeColors],
  );

  const relTypes = useMemo(
    () => Object.entries(state.relationshipTypeColors),
    [state.relationshipTypeColors],
  );

  const allChips = useMemo(
    () => [
      ...nodeTypes.map(([type, color]) => ({ kind: 'node' as const, type, color })),
      ...relTypes.map(([type, color]) => ({ kind: 'rel' as const, type, color })),
    ],
    [nodeTypes, relTypes],
  );

  const activeNodes = state.currentFilters.nodeTypes;
  const activeRels = state.currentFilters.relationshipTypes;

  const isActive = (chip: (typeof allChips)[number]) =>
    chip.kind === 'node' ? activeNodes.has(chip.type) : activeRels.has(chip.type);

  // Measure overflow
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver(() => {
      const children = Array.from(container.querySelectorAll('[data-chip]'));
      let count = 0;
      for (const child of children) {
        const rect = child.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();
        if (rect.right <= containerRect.right + 4) {
          count++;
        } else {
          break;
        }
      }
      setVisibleCount(count < allChips.length ? count : null);
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, [allChips.length]);

  function toggleChip(chip: (typeof allChips)[number]) {
    const nodes = new Set(activeNodes);
    const rels = new Set(activeRels);

    if (chip.kind === 'node') {
      if (nodes.has(chip.type)) nodes.delete(chip.type);
      else nodes.add(chip.type);
    } else {
      if (rels.has(chip.type)) rels.delete(chip.type);
      else rels.add(chip.type);
    }

    dispatch({ type: 'SET_FILTERS', nodeTypes: nodes, relationshipTypes: rels });
  }

  function activateAll() {
    const nodes = new Set(nodeTypes.map(([t]) => t));
    const rels = new Set(relTypes.map(([t]) => t));
    dispatch({ type: 'SET_FILTERS', nodeTypes: nodes, relationshipTypes: rels });
  }

  function deactivateAll() {
    dispatch({ type: 'SET_FILTERS', nodeTypes: new Set(), relationshipTypes: new Set() });
  }

  return (
    <div className="flex items-center gap-1.5 px-3 py-1.5 overflow-y-auto">
      {/* All / None */}
      <button
        type="button"
        onClick={activateAll}
        className="btn btn-xs btn-primary text-2xs shrink-0"
      >
        All
      </button>
      <button
        type="button"
        onClick={deactivateAll}
        className="btn btn-xs btn-ghost text-2xs shrink-0"
      >
        None
      </button>

      <div className="w-px h-3.5 bg-base-300 shrink-0" />

      {/* Chips container */}
      <div ref={containerRef} className="flex items-center gap-1.5 flex-wrap min-w-0">
        {allChips.map((chip) => {
          const active = isActive(chip);
          return (
            <button
              key={`${chip.kind}-${chip.type}`}
              type="button"
              data-chip
              onClick={() => toggleChip(chip)}
              className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-2xs whitespace-nowrap transition-opacity shrink-0 ${
                active
                  ? 'bg-base-300 text-base-content'
                  : 'bg-base-300/40 text-base-content/30'
              }`}
            >
              <span
                className="size-2 rounded-full shrink-0 transition-opacity"
                style={{
                  backgroundColor: chip.color,
                  opacity: active ? 1 : 0.3,
                }}
              />
              {chip.type}
            </button>
          );
        })}
      </div>

      {/* Overflow indicator */}
      {visibleCount !== null && (
        <span className="text-2xs text-base-content/40 shrink-0 pl-1">
          +{allChips.length - visibleCount} more
        </span>
      )}
    </div>
  );
});
