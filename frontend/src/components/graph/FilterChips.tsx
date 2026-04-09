import { memo, useMemo, useRef, useState, useEffect, useCallback } from 'react';
import { CheckIcon, MinusIcon } from '@heroicons/react/20/solid';
import { StopIcon } from '@heroicons/react/24/outline';
import { useApp } from '../../context/AppContext';

// ── Smart Select ────────────────────────────────────────────────────────────

type SmartSelectState = 'all' | 'none' | 'partial';

interface SmartSelectProps {
  selectState: SmartSelectState;
  onAll: () => void;
  onNone: () => void;
  onOntology: () => void;
  onEdges: () => void;
}

function SmartSelect({ selectState, onAll, onNone, onOntology, onEdges }: SmartSelectProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    function handleClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [open]);

  const handleSelect = useCallback(
    (action: () => void) => {
      action();
      setOpen(false);
    },
    [],
  );

  return (
    <div ref={containerRef} className="relative flex items-center shrink-0">
      {/* Drop-up menu */}
      {open && (
        <div className="absolute bottom-full left-0 mb-1 bg-base-100 border border-base-300 rounded-lg shadow-lg z-20 min-w-max">
          <button
            type="button"
            onClick={() => handleSelect(onAll)}
            className="flex items-center gap-2 w-full text-left px-3 py-1.5 text-xs hover:bg-base-200 transition-colors first:rounded-t-lg text-base-content"
          >
            <CheckIcon className="size-3.5 text-primary" />
            All
          </button>
          <button
            type="button"
            onClick={() => handleSelect(onNone)}
            className="flex items-center gap-2 w-full text-left px-3 py-1.5 text-xs hover:bg-base-200 transition-colors text-base-content"
          >
            <StopIcon className="size-3.5 text-base-content/50" />
            None
          </button>
          <button
            type="button"
            onClick={() => handleSelect(onOntology)}
            className="flex items-center gap-2 w-full text-left px-3 py-1.5 text-xs hover:bg-base-200 transition-colors text-base-content"
          >
            <span className="size-3.5 rounded-full bg-primary/20 border border-primary/40 shrink-0" />
            Ontology
          </button>
          <button
            type="button"
            onClick={() => handleSelect(onEdges)}
            className="flex items-center gap-2 w-full text-left px-3 py-1.5 text-xs hover:bg-base-200 transition-colors last:rounded-b-lg text-base-content"
          >
            <span className="size-3.5 rounded-sm bg-secondary/20 border border-secondary/40 shrink-0" />
            Edges
          </button>
        </div>
      )}

      {/* Trigger: checkbox icon */}
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex items-center justify-center size-6 rounded hover:bg-base-200 transition-colors text-base-content/70 hover:text-base-content"
        title="Select filter preset"
      >
        {selectState === 'all' && <CheckIcon className="size-4 text-primary" />}
        {selectState === 'none' && <StopIcon className="size-4 text-base-content/40" />}
        {selectState === 'partial' && <MinusIcon className="size-4 text-base-content/60" />}
      </button>
    </div>
  );
}

// ── Chip ─────────────────────────────────────────────────────────────────────

interface ChipProps {
  type: string;
  color: string;
  active: boolean;
  highlighted: boolean;
  onClick: () => void;
}

function Chip({ type, color, active, highlighted, onClick }: ChipProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        'flex items-center gap-1 px-2 py-0.5 rounded-full text-2xs whitespace-nowrap transition-opacity shrink-0',
        active ? 'bg-base-300 text-base-content' : 'bg-base-300/40 text-base-content/30',
        highlighted ? 'outline outline-1.5 outline-offset-1 outline-[oklch(72%_0.12_200)]' : '',
      ]
        .filter(Boolean)
        .join(' ')}
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
}

function ChipColumn({ placeholder, entries, activeSet, onToggle }: ChipColumnProps) {
  const [query, setQuery] = useState('');

  const filtered = useMemo(
    () => (query.trim() === '' ? entries : entries.filter(([t]) => t.toLowerCase().includes(query.toLowerCase()))),
    [entries, query],
  );

  return (
    <div className="flex flex-col flex-1 min-w-0 min-h-0 gap-1">
      {/* Search input */}
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder={placeholder}
        className="border border-base-300 rounded px-2 py-0.5 text-2xs bg-transparent text-base-content placeholder:text-base-content/30 focus:outline-none focus:border-primary/50 shrink-0"
      />
      {/* Chips */}
      <div className="flex flex-wrap gap-1 overflow-y-auto min-h-0">
        {filtered.map(([type, color]) => (
          <Chip
            key={type}
            type={type}
            color={color}
            active={activeSet.has(type)}
            highlighted={query.trim() !== '' && type.toLowerCase().includes(query.toLowerCase())}
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
    <div className="w-px self-stretch bg-gradient-to-b from-transparent via-base-300 to-transparent shrink-0" />
  );
}

// ── FilterChips (main export) ────────────────────────────────────────────────

export const FilterChips = memo(function FilterChips() {
  const { state, dispatch } = useApp();

  const nodeTypes = useMemo(() => Object.entries(state.nodeTypeColors), [state.nodeTypeColors]);
  const relTypes = useMemo(() => Object.entries(state.relationshipTypeColors), [state.relationshipTypeColors]);

  const activeNodes = state.currentFilters.nodeTypes;
  const activeRels = state.currentFilters.relationshipTypes;

  // Derive smart-select checkbox state
  const selectState = useMemo((): SmartSelectState => {
    const totalTypes = nodeTypes.length + relTypes.length;
    if (totalTypes === 0) return 'none';
    const activeCount = activeNodes.size + activeRels.size;
    if (activeCount === 0) return 'none';
    if (activeCount === totalTypes) return 'all';
    return 'partial';
  }, [nodeTypes.length, relTypes.length, activeNodes.size, activeRels.size]);

  function activateAll() {
    const nodes = new Set(nodeTypes.map(([t]) => t));
    const rels = new Set(relTypes.map(([t]) => t));
    dispatch({ type: 'SET_FILTERS', nodeTypes: nodes, relationshipTypes: rels });
  }

  function deactivateAll() {
    dispatch({ type: 'SET_FILTERS', nodeTypes: new Set(), relationshipTypes: new Set() });
  }

  function activateOntology() {
    const nodes = new Set(nodeTypes.map(([t]) => t));
    dispatch({ type: 'SET_FILTERS', nodeTypes: nodes, relationshipTypes: new Set() });
  }

  function activateEdges() {
    const rels = new Set(relTypes.map(([t]) => t));
    dispatch({ type: 'SET_FILTERS', nodeTypes: new Set(), relationshipTypes: rels });
  }

  function toggleNode(type: string) {
    const nodes = new Set(activeNodes);
    if (nodes.has(type)) nodes.delete(type);
    else nodes.add(type);
    dispatch({ type: 'SET_FILTERS', nodeTypes: nodes, relationshipTypes: new Set(activeRels) });
  }

  function toggleRel(type: string) {
    const rels = new Set(activeRels);
    if (rels.has(type)) rels.delete(type);
    else rels.add(type);
    dispatch({ type: 'SET_FILTERS', nodeTypes: new Set(activeNodes), relationshipTypes: rels });
  }

  return (
    <div className="flex items-stretch gap-2 px-2 py-1.5 min-h-0">
      {/* Smart select checkbox */}
      <SmartSelect
        selectState={selectState}
        onAll={activateAll}
        onNone={deactivateAll}
        onOntology={activateOntology}
        onEdges={activateEdges}
      />

      <VSeparator />

      {/* Nodes column */}
      <ChipColumn
        placeholder="Filter nodes..."
        entries={nodeTypes}
        activeSet={activeNodes}
        onToggle={toggleNode}
      />

      <VSeparator />

      {/* Edges column */}
      <ChipColumn
        placeholder="Filter edges..."
        entries={relTypes}
        activeSet={activeRels}
        onToggle={toggleRel}
      />
    </div>
  );
});
