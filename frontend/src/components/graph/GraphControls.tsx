import {
  ArrowDownTrayIcon,
  ArrowPathIcon,
  MagnifyingGlassIcon,
  MinusIcon,
  PlusIcon,
  XMarkIcon,
} from '@heroicons/react/20/solid';
import { useCallback, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { showError, showSuccess } from '../ui/Notifications';

export function GraphControls() {
  const { state, dispatch, networkRef, initialViewRef } = useApp();
  const [searchTerm, setSearchTerm] = useState('');
  const [matchCount, setMatchCount] = useState<number | null>(null);

  // ── Zoom ────────────────────────────────────────────────────────

  const handleZoomIn = () => {
    const network = networkRef.current;
    if (!network) return;
    network.moveTo({ scale: network.getScale() * 1.2, animation: true });
  };

  const handleZoomOut = () => {
    const network = networkRef.current;
    if (!network) return;
    network.moveTo({ scale: network.getScale() * 0.8, animation: true });
  };

  const handleResetZoom = () => {
    const network = networkRef.current;
    if (!network) return;
    dispatch({ type: 'CLEAR_FILTERS' });
    dispatch({ type: 'CLEAR_HIGHLIGHTED_NODES' });
    setSearchTerm('');
    setMatchCount(null);
    setTimeout(() => {
      if (initialViewRef.current?.position && initialViewRef.current?.scale) {
        network.moveTo({
          position: initialViewRef.current.position,
          scale: initialViewRef.current.scale,
          animation: true,
        });
      } else {
        network.fit({ animation: true });
      }
    }, 100);
  };

  // ── Toggle handlers ─────────────────────────────────────────────

  const handlePhysicsToggle = () => {
    dispatch({ type: 'TOGGLE_PHYSICS' });
    const network = networkRef.current;
    if (network) {
      network.setOptions({ physics: { enabled: !state.physicsEnabled } });
    }
  };

  const handleLabelsToggle = () => {
    const newValue = !state.showEdgeLabels;
    dispatch({ type: 'TOGGLE_EDGE_LABELS' });
    const net = networkRef.current;
    if (!net) return;

    const edges = net.body.data.edges;
    const allEdges = edges.get() as Array<Record<string, unknown>>;
    edges.update(
      allEdges.map((edge: Record<string, unknown>) => ({
        id: edge.id,
        font: { ...(edge.font as Record<string, unknown>), size: newValue ? 11 : 0 },
      })),
    );

    const nodes = net.body.data.nodes;
    const allNodes = nodes.get() as Array<Record<string, unknown>>;
    nodes.update(
      allNodes.map((node: Record<string, unknown>) => ({
        id: node.id,
        font: { ...(node.font as Record<string, unknown>), size: newValue ? 11 : 0 },
      })),
    );
  };

  const handleSizeMetricToggle = () => {
    const newMetric = state.nodeSizeMetric === 'degree' ? 'uniform' : 'degree';
    dispatch({ type: 'SET_NODE_SIZE_METRIC', metric: newMetric });

    const net = networkRef.current;
    if (!net) return;

    const nodes = net.body.data.nodes;
    const allNodes = nodes.get() as Array<Record<string, unknown>>;
    const UNIFORM_WIDTH = 60;

    nodes.update(
      allNodes.map((node: Record<string, unknown>) => {
        const w =
          newMetric === 'uniform' ? UNIFORM_WIDTH : ((node._baseWidth as number) ?? UNIFORM_WIDTH);
        return { id: node.id, widthConstraint: { minimum: w, maximum: w } };
      }),
    );
  };

  // ── Search ──────────────────────────────────────────────────────

  const performSearch = useCallback(
    (term: string) => {
      const network = networkRef.current;
      if (!network) return;

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
        setMatchCount(null);
        network.redraw();
        return;
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

      setMatchCount(matched.size);
      network.redraw();
    },
    [networkRef],
  );

  const handleSearchInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setSearchTerm(value);
    performSearch(value);
  };

  const handleSearchClear = () => {
    setSearchTerm('');
    performSearch('');
  };

  // ── Export ──────────────────────────────────────────────────────

  const handleExportPNG = () => {
    const network = networkRef.current;
    if (!network) {
      showError(dispatch, 'Please load a knowledge graph first');
      return;
    }
    try {
      const canvas = network.canvas.frame.canvas;
      const link = document.createElement('a');
      link.download = `kg_${state.currentKGName || 'graph'}_${new Date().toISOString().slice(0, 10)}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      showError(dispatch, `PNG export failed: ${msg}`);
    }
  };

  const handleExportJSON = () => {
    if (!state.graphData) {
      showError(dispatch, 'Please load a knowledge graph first');
      return;
    }
    try {
      const dataStr = JSON.stringify(state.graphData, null, 2);
      const dataUri = `data:application/json;charset=utf-8,${encodeURIComponent(dataStr)}`;
      const link = document.createElement('a');
      link.setAttribute('href', dataUri);
      link.setAttribute('download', `knowledge_graph_${new Date().toISOString().split('T')[0]}.json`);
      link.click();
      showSuccess(dispatch, 'Graph exported successfully!');
    } catch {
      showError(dispatch, 'Failed to export graph data');
    }
  };

  // ── Render ──────────────────────────────────────────────────────

  const toggleBtn = (active: boolean) =>
    `btn btn-xs min-w-[5.5rem] ${active ? 'bg-base-300 text-[color:oklch(62%_0.10_270)]' : 'bg-base-300 text-base-content/60'}`;

  return (
    <div className="flex flex-wrap items-center gap-1 px-2 py-1">
      {/* Left: Zoom controls */}
      <div className="join shrink-0">
        <button
          type="button"
          className="btn btn-ghost btn-xs join-item"
          onClick={handleZoomIn}
          aria-label="Zoom in"
          title="Zoom in"
        >
          <PlusIcon className="size-4" aria-hidden="true" />
        </button>
        <button
          type="button"
          className="btn btn-ghost btn-xs join-item"
          onClick={handleZoomOut}
          aria-label="Zoom out"
          title="Zoom out"
        >
          <MinusIcon className="size-4" aria-hidden="true" />
        </button>
        <button
          type="button"
          className="btn btn-ghost btn-xs join-item"
          onClick={handleResetZoom}
          aria-label="Reset view"
          title="Reset view"
        >
          <ArrowPathIcon className="size-4" aria-hidden="true" />
        </button>
      </div>

      {/* Divider */}
      <div className="w-px h-5 bg-base-300/50 shrink-0" />

      {/* Toggles */}
      <button type="button" className={toggleBtn(state.physicsEnabled)} onClick={handlePhysicsToggle}>
        Physics {state.physicsEnabled ? 'ON' : 'OFF'}
      </button>

      <button
        type="button"
        className={toggleBtn(state.showEdgeLabels)}
        onClick={handleLabelsToggle}
      >
        Labels {state.showEdgeLabels ? 'ON' : 'OFF'}
      </button>

      {/* Degree / Uniform toggle */}
      <div className="flex rounded-md bg-base-300 shrink-0 overflow-hidden">
        <button
          type="button"
          className={`btn btn-xs border-none shadow-none ${state.nodeSizeMetric === 'degree' ? 'text-[color:oklch(62%_0.10_270)]' : 'text-base-content/60'}`}
          onClick={() => state.nodeSizeMetric !== 'degree' && handleSizeMetricToggle()}
        >
          Degree
        </button>
        <div className="w-px bg-base-content/10 my-1" />
        <button
          type="button"
          className={`btn btn-xs border-none shadow-none ${state.nodeSizeMetric !== 'degree' ? 'text-[color:oklch(62%_0.10_270)]' : 'text-base-content/60'}`}
          onClick={() => state.nodeSizeMetric === 'degree' && handleSizeMetricToggle()}
        >
          Uniform
        </button>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Center: Search */}
      <div className="relative flex-1 min-w-0 max-w-[160px]">
        <MagnifyingGlassIcon className="absolute left-2 top-1/2 -translate-y-1/2 size-3.5 text-base-content/40 pointer-events-none" />
        <input
          type="text"
          className="input input-bordered input-xs w-full pl-7 pr-14"
          placeholder="Search nodes..."
          value={searchTerm}
          onChange={handleSearchInput}
          autoComplete="off"
        />
        {searchTerm && (
          <button
            type="button"
            className="absolute right-7 top-1/2 -translate-y-1/2 opacity-50 hover:opacity-100 transition-opacity"
            onClick={handleSearchClear}
            aria-label="Clear search"
          >
            <XMarkIcon className="size-3.5" aria-hidden="true" />
          </button>
        )}
        {matchCount !== null && matchCount > 0 && (
          <span className="absolute right-1.5 top-1/2 -translate-y-1/2 text-2xs font-semibold text-accent pointer-events-none">
            {matchCount}
          </span>
        )}
      </div>

      {/* Highlight indicator */}
      {state.highlightedNodes.size > 0 && (
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[color:oklch(85%_0.18_85_/_0.12)] shrink-0">
          <div className="size-1.5 rounded-full bg-[color:oklch(85%_0.18_85)]" />
          <span className="text-2xs font-medium text-[color:oklch(85%_0.18_85)]">
            {state.highlightedNodes.size} highlighted
          </span>
          <button
            type="button"
            className="opacity-60 hover:opacity-100 transition-opacity"
            onClick={() => dispatch({ type: 'CLEAR_HIGHLIGHTED_NODES' })}
            aria-label="Clear highlights"
          >
            <XMarkIcon className="size-3" aria-hidden="true" />
          </button>
        </div>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Right: Export */}
      <div className="flex gap-1 ml-auto shrink-0">
        <button
          type="button"
          className="btn btn-xs bg-base-300 text-base-content/60 shrink-0"
          onClick={handleExportPNG}
          title="Export as PNG"
          aria-label="Export as PNG"
        >
          <ArrowDownTrayIcon className="size-3.5 inline" aria-hidden="true" /> PNG
        </button>
        <button
          type="button"
          className="btn btn-xs bg-base-300 text-base-content/60 shrink-0"
          onClick={handleExportJSON}
          title="Export as JSON"
          aria-label="Export as JSON"
        >
          <ArrowDownTrayIcon className="size-3.5 inline" aria-hidden="true" /> JSON
        </button>
      </div>
    </div>
  );
}
