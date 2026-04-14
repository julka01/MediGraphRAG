import { useCallback, useEffect, useState } from 'react';
import { MagnifyingGlassIcon, XMarkIcon } from '@heroicons/react/20/solid';
import { useApp } from '../../context/AppContext';

export function applySearchToNetwork(networkRef: { current: vis.Network | null }, term: string): number {
  const network = networkRef.current;
  if (!network) return 0;

  const nodeDS = network.body.data.nodes;
  const edgeDS = network.body.data.edges;
  const t = (term || '').toLowerCase().trim();

  if (!t) {
    nodeDS.update(nodeDS.get().map((n: Record<string, unknown>) => ({ id: n.id, opacity: 1, hidden: false })));
    edgeDS.update(edgeDS.get().map((e: Record<string, unknown>) => ({ id: e.id, opacity: 1, hidden: false })));
    network.redraw();
    return 0;
  }

  const matched = new Set<string | number>();
  const nodeUpdates = nodeDS.get().map((node: Record<string, unknown>) => {
    const hit =
      ((node.label as string) || '').toLowerCase().includes(t) ||
      ((node.title as string) || '').toLowerCase().includes(t) ||
      JSON.stringify(node.properties || {})
        .toLowerCase()
        .includes(t);
    if (hit) matched.add(node.id as string | number);
    return { id: node.id, hidden: false, opacity: hit ? 1 : 0.1 };
  });
  nodeDS.update(nodeUpdates);

  const edgeUpdates = edgeDS.get().map((edge: Record<string, unknown>) => ({
    id: edge.id,
    hidden: false,
    opacity: matched.has(edge.from as string | number) || matched.has(edge.to as string | number) ? 1 : 0.06,
  }));
  edgeDS.update(edgeUpdates);

  network.redraw();
  return matched.size;
}

export function TopBar() {
  const { state, dispatch, networkRef } = useApp();
  const [matchCount, setMatchCount] = useState(0);

  const performSearch = useCallback(
    (term: string) => {
      setMatchCount(applySearchToNetwork(networkRef, term));
    },
    [networkRef],
  );

  // Re-count matches after filters rebuild the graph
  useEffect(() => {
    if (state.searchTerm.trim()) {
      // Defer so the graph has finished rebuilding after filter change
      const id = requestAnimationFrame(() => {
        setMatchCount(applySearchToNetwork(networkRef, state.searchTerm));
      });
      return () => cancelAnimationFrame(id);
    }
  }, [state.currentFilters, networkRef, state.searchTerm]);

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

  return (
    <div className="shrink-0 border-b border-base-content/10 bg-base-100/55 backdrop-blur-xl">
      <div className="flex h-14 items-center gap-3 px-4">
        <div className="hidden min-w-[8rem] lg:block">
          <p className="text-[0.65rem] font-medium uppercase tracking-[0.18em] text-base-content/45">Search</p>
          <p className="mt-1 text-sm font-medium text-base-content/72">Find nodes and traces</p>
        </div>
        <div className="relative min-w-0 flex-1">
          <MagnifyingGlassIcon
            className="absolute left-2 top-1/2 -translate-y-1/2 size-4 text-base-content/40 pointer-events-none"
            aria-hidden="true"
          />
          <input
            type="text"
            aria-label="Search nodes"
            className="h-10 w-full rounded-xl border border-base-content/12 bg-base-100/65 pl-8 pr-4 text-sm shadow-sm transition-colors placeholder:text-base-content/30 focus-visible:border-primary/50 focus-visible:outline-none"
            placeholder="Search nodes, properties, and connected entities"
            value={state.searchTerm}
            onChange={handleSearchInput}
            autoComplete="off"
          />
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {state.searchTerm.trim() && matchCount > 0 && (
            <span className="badge badge-soft badge-accent badge-sm gap-1 text-2xs pr-1">
              {matchCount} found
              <button
                type="button"
                className="opacity-60 transition-opacity hover:opacity-100"
                onClick={handleSearchClear}
                aria-label="Clear search"
              >
                <XMarkIcon className="size-3" aria-hidden="true" />
              </button>
            </span>
          )}
          {state.searchTerm.trim() && matchCount === 0 && (
            <button
              type="button"
              className="btn btn-ghost btn-xs rounded-full text-base-content/50 hover:text-base-content/85"
              onClick={handleSearchClear}
              aria-label="Clear search"
            >
              <XMarkIcon className="size-4" aria-hidden="true" />
            </button>
          )}
          {state.highlightedCount > 0 && (
            <span className="badge badge-sm gap-1 text-2xs pr-1 bg-graph-highlight/12 text-graph-highlight">
              {state.highlightedCount} highlighted
              <button
                type="button"
                className="opacity-60 transition-opacity hover:opacity-100"
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
  );
}
