import {
  ArrowDownTrayIcon,
  ArrowPathIcon,
  MinusIcon,
  PlusIcon,
} from '@heroicons/react/20/solid';
import { useApp } from '../../context/AppContext';
import { showError, showSuccess } from '../ui/Notifications';

export function GraphControls() {
  const { state, dispatch, networkRef, initialViewRef } = useApp();

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
    // CLEAR_HIGHLIGHTED_NODES triggers a full graph rebuild in useGraph,
    // which resets any search-applied opacity on nodes/edges.
    dispatch({ type: 'CLEAR_HIGHLIGHTED_NODES' });
    dispatch({ type: 'SET_SEARCH_TERM', payload: '' });
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
    `btn btn-xs shadow-none min-w-[5.5rem] ${active ? 'bg-base-300 text-[color:oklch(62%_0.10_270)]' : 'bg-base-300 text-base-content/60'}`;

  return (
    <div className="flex flex-wrap items-center gap-1 px-2 py-1">
      {/* Left: Zoom controls */}
      <div className="join shrink-0">
        <button type="button" className="btn btn-xs shadow-none bg-base-300 text-base-content/60 join-item" onClick={handleZoomIn} aria-label="Zoom in" title="Zoom in">
          <PlusIcon className="size-4" aria-hidden="true" />
        </button>
        <button type="button" className="btn btn-xs shadow-none bg-base-300 text-base-content/60 join-item" onClick={handleZoomOut} aria-label="Zoom out" title="Zoom out">
          <MinusIcon className="size-4" aria-hidden="true" />
        </button>
        <button type="button" className="btn btn-xs shadow-none bg-base-300 text-base-content/60 join-item" onClick={handleResetZoom} aria-label="Reset view" title="Reset view">
          <ArrowPathIcon className="size-4" aria-hidden="true" />
        </button>
      </div>

      {/* Divider */}
      <div className="w-px h-5 bg-base-300/50 shrink-0" />

      {/* Toggles */}
      <button type="button" className={toggleBtn(state.physicsEnabled)} onClick={handlePhysicsToggle}>
        Physics {state.physicsEnabled ? 'ON' : 'OFF'}
      </button>

      <button type="button" className={toggleBtn(state.showEdgeLabels)} onClick={handleLabelsToggle}>
        Labels {state.showEdgeLabels ? 'ON' : 'OFF'}
      </button>

      {/* Uniform / Degree toggle */}
      <div className="flex shrink-0 isolate">
        <button
          type="button"
          className={`btn btn-xs shadow-none min-w-0 px-2 rounded-r-none relative hover:z-10 ${state.nodeSizeMetric !== 'degree' ? 'bg-base-300 text-[color:oklch(62%_0.10_270)]' : 'bg-base-300 text-base-content/60'}`}
          onClick={() => state.nodeSizeMetric === 'degree' && handleSizeMetricToggle()}
        >
          Uniform
        </button>
        <button
          type="button"
          className={`btn btn-xs shadow-none min-w-0 px-2 rounded-l-none -ml-px relative hover:z-10 ${state.nodeSizeMetric === 'degree' ? 'bg-base-300 text-[color:oklch(62%_0.10_270)]' : 'bg-base-300 text-base-content/60'}`}
          onClick={() => state.nodeSizeMetric !== 'degree' && handleSizeMetricToggle()}
        >
          Degree
        </button>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Right: Export */}
      <div className="join ml-auto shrink-0">
        <button type="button" className="btn btn-xs shadow-none bg-base-300 text-base-content/60 join-item w-20" onClick={handleExportPNG} title="Export as PNG" aria-label="Export as PNG">
          <ArrowDownTrayIcon className="size-3.5 inline" aria-hidden="true" /> PNG
        </button>
        <button type="button" className="btn btn-xs shadow-none bg-base-300 text-base-content/60 join-item w-20" onClick={handleExportJSON} title="Export as JSON" aria-label="Export as JSON">
          <ArrowDownTrayIcon className="size-3.5 inline" aria-hidden="true" /> JSON
        </button>
      </div>
    </div>
  );
}
