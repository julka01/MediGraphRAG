import { memo, useMemo, useState } from 'react';
import { XMarkIcon } from '@heroicons/react/20/solid';
import { useApp } from '../../context/AppContext';

// ── Select-all checkbox state ───────────────────────────────────────────────

type CheckState = 'all' | 'none' | 'partial';

// ── Chip ─────────────────────────────────────────────────────────────────────

interface ChipProps {
  type: string;
  color: string;
  active: boolean;
  onClick: () => void;
}

function Chip({ type, color, active, onClick }: ChipProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-2xs whitespace-nowrap transition-opacity shrink-0 ${
        active ? 'bg-base-300 text-base-content' : 'bg-base-300/40 text-base-content/30'
      }`}
    >
      <span
        className="size-2 rounded-full shrink-0 transition-opacity"
        style={{ backgroundColor: color, opacity: active ? 1 : 0.3 }}
      />
      {type}
    </button>
  );
}

// ── Column ───────────────────────────────────────────────────────────────────

interface ChipColumnProps {
  placeholder: string;
  entries: [string, string][];
  activeSet: Set<string>;
  onToggle: (type: string) => void;
  onAll: () => void;
  onNone: () => void;
}

function ChipColumn({ placeholder, entries, activeSet, onToggle, onAll, onNone }: ChipColumnProps) {
  const [query, setQuery] = useState('');

  const filtered = useMemo(
    () => (query.trim() === '' ? entries : entries.filter(([t]) => t.toLowerCase().includes(query.toLowerCase()))),
    [entries, query],
  );

  const checkState = useMemo((): CheckState => {
    if (entries.length === 0 || activeSet.size === 0) return 'none';
    if (activeSet.size === entries.length) return 'all';
    return 'partial';
  }, [entries.length, activeSet.size]);

  const isChecked = checkState === 'all';
  const isIndeterminate = checkState === 'partial';

  return (
    <div className="flex flex-col flex-1 min-w-0 min-h-0 gap-1.5">
      {/* Sticky header: checkbox + search */}
      <div className="flex items-center gap-1.5 shrink-0">
        <input
          type="checkbox"
          checked={isChecked}
          ref={(el) => { if (el) el.indeterminate = isIndeterminate; }}
          onChange={() => (isChecked || isIndeterminate) ? onNone() : onAll()}
          className="checkbox rounded-sm border-base-content/30 [--chkbg:transparent] checked:bg-transparent indeterminate:bg-transparent size-[1.5rem] shrink-0 translate-y-px shadow-none text-base-content/60"
          title={isChecked ? 'Deselect all' : 'Select all'}
        />
        <div className="relative flex-1 min-w-0">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={placeholder}
            className="w-full border border-base-content/20 rounded px-2 py-1 text-2xs bg-transparent text-base-content placeholder:text-base-content/30 focus:outline-none focus:border-primary/50"
          />
          {query && (
            <button
              type="button"
              className="absolute inset-y-0 right-1.5 my-auto flex items-center justify-center size-3.5 translate-y-px opacity-50 hover:opacity-100 transition-opacity"
              onClick={() => setQuery('')}
              aria-label="Clear filter"
            >
              <XMarkIcon className="size-3.5" aria-hidden="true" />
            </button>
          )}
        </div>
      </div>
      {/* Scrollable chips */}
      <div className="flex flex-wrap gap-1 overflow-y-auto min-h-0">
        {filtered.map(([type, color]) => (
          <Chip
            key={type}
            type={type}
            color={color}
            active={activeSet.has(type)}
            onClick={() => onToggle(type)}
          />
        ))}
      </div>
    </div>
  );
}

// ── Separator ────────────────────────────────────────────────────────────────

function VSeparator() {
  return (
    <div className="w-px self-stretch bg-gradient-to-b from-transparent via-base-content/20 to-transparent shrink-0" />
  );
}

// ── FilterChips (main export) ────────────────────────────────────────────────

export const FilterChips = memo(function FilterChips() {
  const { state, dispatch } = useApp();

  const nodeTypes = useMemo(() => Object.entries(state.nodeTypeColors), [state.nodeTypeColors]);
  const relTypes = useMemo(() => Object.entries(state.relationshipTypeColors), [state.relationshipTypeColors]);

  const rawActiveNodes = state.currentFilters.nodeTypes;
  const rawActiveRels = state.currentFilters.relationshipTypes;

  // null = all selected (uninitialized); materialize for display
  const activeNodes = useMemo(
    () => rawActiveNodes ?? new Set(nodeTypes.map(([t]) => t)),
    [rawActiveNodes, nodeTypes],
  );
  const activeRels = useMemo(
    () => rawActiveRels ?? new Set(relTypes.map(([t]) => t)),
    [rawActiveRels, relTypes],
  );

  function selectAllNodes() {
    dispatch({ type: 'SET_FILTERS', nodeTypes: new Set(nodeTypes.map(([t]) => t)) });
  }
  function selectNoNodes() {
    dispatch({ type: 'SET_FILTERS', nodeTypes: new Set() });
  }
  function selectAllRels() {
    dispatch({ type: 'SET_FILTERS', relationshipTypes: new Set(relTypes.map(([t]) => t)) });
  }
  function selectNoRels() {
    dispatch({ type: 'SET_FILTERS', relationshipTypes: new Set() });
  }

  function toggleNode(type: string) {
    const nodes = new Set(activeNodes);
    if (nodes.has(type)) nodes.delete(type);
    else nodes.add(type);
    dispatch({ type: 'SET_FILTERS', nodeTypes: nodes });
  }

  function toggleRel(type: string) {
    const rels = new Set(activeRels);
    if (rels.has(type)) rels.delete(type);
    else rels.add(type);
    dispatch({ type: 'SET_FILTERS', relationshipTypes: rels });
  }

  return (
    <div className="flex items-stretch gap-2 px-2 py-1.5 min-h-0 h-full">
      {/* Nodes column */}
      <ChipColumn
        placeholder="Filter nodes..."
        entries={nodeTypes}
        activeSet={activeNodes}
        onToggle={toggleNode}
        onAll={selectAllNodes}
        onNone={selectNoNodes}
      />

      <VSeparator />

      {/* Edges column */}
      <ChipColumn
        placeholder="Filter edges..."
        entries={relTypes}
        activeSet={activeRels}
        onToggle={toggleRel}
        onAll={selectAllRels}
        onNone={selectNoRels}
      />
    </div>
  );
});
