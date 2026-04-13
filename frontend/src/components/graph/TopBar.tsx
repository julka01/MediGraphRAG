import { useCallback, useRef } from 'react';
import { MagnifyingGlassIcon, XMarkIcon } from '@heroicons/react/20/solid';
import { useApp } from '../../context/AppContext';

export function applySearchToNetwork(
  networkRef: { current: vis.Network | null },
  term: string,
): number {
  const network = networkRef.current;
  if (!network) return 0;

  const nodeDS = network.body.data.nodes;
  const edgeDS = network.body.data.edges;
  const t = (term || '').toLowerCase().trim();

  if (!t) {
    nodeDS.update(
      nodeDS.get().map((n: Record<string, unknown>) => ({ id: n.id, opacity: 1, hidden: false })),
    );
    edgeDS.update(
      edgeDS.get().map((e: Record<string, unknown>) => ({ id: e.id, opacity: 1, hidden: false })),
    );
    network.redraw();
    return 0;
  }

  const matched = new Set<string | number>();
  const nodeUpdates = nodeDS.get().map((node: Record<string, unknown>) => {
    const hit =
      ((node.label as string) || '').toLowerCase().includes(t) ||
      ((node.title as string) || '').toLowerCase().includes(t) ||
      JSON.stringify(node.properties || {}).toLowerCase().includes(t);
    if (hit) matched.add(node.id as string | number);
    return { id: node.id, hidden: false, opacity: hit ? 1 : 0.1 };
  });
  nodeDS.update(nodeUpdates);

  const edgeUpdates = edgeDS.get().map((edge: Record<string, unknown>) => ({
    id: edge.id,
    hidden: false,
    opacity:
      matched.has(edge.from as string | number) || matched.has(edge.to as string | number)
        ? 1
        : 0.06,
  }));
  edgeDS.update(edgeUpdates);

  network.redraw();
  return matched.size;
}

export function TopBar() {
  const { state, dispatch, networkRef } = useApp();
  const matchCountRef = useRef(0);

  const performSearch = useCallback(
    (term: string) => {
      matchCountRef.current = applySearchToNetwork(networkRef, term);
    },
    [networkRef],
  );

  const handleSearchInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    dispatch({ type: 'SET_SEARCH_TERM', payload: value });
    performSearch(value);
  };

  const handleSearchClear = () => {
    dispatch({ type: 'SET_SEARCH_TERM', payload: '' });
    performSearch('');
  };

  const handleHighlightsClear = () => {
    dispatch({ type: 'CLEAR_HIGHLIGHTED_NODES' });
  };

  const matchCount = state.searchTerm.trim() ? matchCountRef.current : 0;

  return (
    <div className="bg-base-200 shrink-0 h-12">
      <div className="flex items-center px-3 h-full">
        <div className="relative flex-1 min-w-0">
          <MagnifyingGlassIcon className="absolute left-2 top-1/2 -translate-y-1/2 size-4 text-base-content/40 pointer-events-none" aria-hidden="true" />
          <input
            type="text"
            aria-label="Search nodes"
            className="w-full h-8 pl-7 pr-4 rounded-lg border border-base-content/20 bg-transparent text-sm placeholder:text-base-content/30 focus-visible:outline-none focus-visible:border-primary/50 transition-colors"
            placeholder="Search nodes…"
            value={state.searchTerm}
            onChange={handleSearchInput}
            autoComplete="off"
          />
          {/* Badges container — inside input, right side */}
          <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1.5">
            {/* Search match count badge */}
            {state.searchTerm.trim() && matchCount > 0 && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-accent/15 text-accent text-2xs font-semibold">
                {matchCount} found
                <button
                  type="button"
                  className="opacity-60 hover:opacity-100 transition-opacity"
                  onClick={handleSearchClear}
                  aria-label="Clear search"
                >
                  <XMarkIcon className="size-3" aria-hidden="true" />
                </button>
              </span>
            )}
            {/* Clear search X (when searching but no matches) */}
            {state.searchTerm.trim() && matchCount === 0 && (
              <button
                type="button"
                className="opacity-50 hover:opacity-100 transition-opacity"
                onClick={handleSearchClear}
                aria-label="Clear search"
              >
                <XMarkIcon className="size-4" aria-hidden="true" />
              </button>
            )}
            {/* Highlighted nodes badge */}
            {state.highlightedCount > 0 && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[color:oklch(85%_0.18_85_/_0.12)] text-[color:oklch(85%_0.18_85)] text-2xs font-semibold">
                {state.highlightedCount} highlighted
                <button
                  type="button"
                  className="opacity-60 hover:opacity-100 transition-opacity"
                  onClick={handleHighlightsClear}
                  aria-label="Clear highlights"
                >
                  <XMarkIcon className="size-3" aria-hidden="true" />
                </button>
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
