import { ArrowDownTrayIcon, ArrowPathIcon, MinusIcon, PlusIcon } from '@heroicons/react/20/solid';
import { useApp } from '../../context/AppContext';
import type { NodeSizeMetric } from '../../types/app';
import { showError, showSuccess } from '../ui/Notifications';
import { ToolbarButton } from './ToolbarButton';

function normalizeGraphValue(value: unknown): string {
  return String(value ?? '')
    .trim()
    .toLowerCase();
}

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

  const handleReflow = () => {
    const network = networkRef.current;
    if (!network) return;
    network.setOptions({
      physics: {
        enabled: true,
        stabilization: {
          enabled: true,
          iterations: 220,
          updateInterval: 25,
        },
      },
    });
    network.stabilize(220);
  };

  const handlePhysicsToggle = () => {
    const network = networkRef.current;
    const nextEnabled = !state.physicsEnabled;
    dispatch({ type: 'SET_PHYSICS', enabled: nextEnabled });
    if (!network) return;

    network.setOptions({
      physics: {
        enabled: nextEnabled,
        stabilization: false,
      },
    });

    if (nextEnabled) {
      network.startSimulation();
    } else {
      network.stopSimulation();
    }
  };

  const handleFocusHighlights = () => {
    const network = networkRef.current;
    if (!network) return;
    if (state.highlightedNodes.size === 0) {
      showError(dispatch, 'Highlight some graph entities first');
      return;
    }

    const highlighted = new Set([...state.highlightedNodes].map(normalizeGraphValue));
    const nodes = network.body.data.nodes.get() as Array<Record<string, unknown>>;
    const targetNodeIds = nodes
      .filter((node) => {
        const props = (node.properties as Record<string, unknown> | undefined) ?? {};
        const candidates = [node.label, node.originalId, props.name, props.id, props.title]
          .filter(Boolean)
          .map(normalizeGraphValue);
        return candidates.some((candidate) => highlighted.has(candidate));
      })
      .map((node) => Number(node.id))
      .filter(Number.isFinite);

    if (targetNodeIds.length === 0) {
      showError(dispatch, 'No highlighted nodes are visible in the current graph view');
      return;
    }

    network.fit({
      nodes: targetNodeIds,
      animation: { duration: 350, easingFunction: 'easeInOutQuad' },
    });
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

  const handleSizeMetricChange = (newMetric: NodeSizeMetric) => {
    if (newMetric === state.nodeSizeMetric) return;
    dispatch({ type: 'SET_NODE_SIZE_METRIC', metric: newMetric });

    const net = networkRef.current;
    if (!net) return;

    const nodes = net.body.data.nodes;
    const allNodes = nodes.get() as Array<Record<string, unknown>>;
    const metricWidthKey: Record<NodeSizeMetric, string> = {
      uniform: '_uniformWidth',
      degree: '_degreeWidth',
      inDegree: '_inDegreeWidth',
      outDegree: '_outDegreeWidth',
      pageRank: '_pageRankWidth',
    };
    const widthKey = metricWidthKey[newMetric];

    nodes.update(
      allNodes.map((node: Record<string, unknown>) => {
        const w = (node[widthKey] as number | undefined) ?? 60;
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

  return (
    <div className="flex flex-wrap items-center gap-2 px-3 py-2">
      {/* Zoom controls */}
      <div className="flex shrink-0 items-center gap-1 rounded-2xl border border-base-content/10 bg-base-100/60 p-1">
        <ToolbarButton onClick={handleZoomIn} aria-label="Zoom in" title="Zoom in">
          <PlusIcon className="size-4" aria-hidden="true" />
        </ToolbarButton>
        <ToolbarButton onClick={handleZoomOut} aria-label="Zoom out" title="Zoom out">
          <MinusIcon className="size-4" aria-hidden="true" />
        </ToolbarButton>
        <ToolbarButton onClick={handleResetZoom} aria-label="Reset view" title="Reset view">
          <ArrowPathIcon className="size-4" aria-hidden="true" />
        </ToolbarButton>
      </div>

      <div className="hidden text-[0.65rem] font-medium uppercase tracking-[0.18em] text-base-content/40 lg:block">
        View
      </div>

      {/* Toggles */}
      <div className="flex shrink-0 items-center gap-1 rounded-2xl border border-base-content/10 bg-base-100/60 p-1">
        <ToolbarButton active={state.physicsEnabled} className="min-w-[6rem]" onClick={handlePhysicsToggle}>
          Physics {state.physicsEnabled ? 'ON' : 'OFF'}
        </ToolbarButton>

        <ToolbarButton className="min-w-[5.4rem]" onClick={handleReflow}>
          Reflow
        </ToolbarButton>

        <ToolbarButton active={state.showEdgeLabels} className="min-w-[5.5rem]" onClick={handleLabelsToggle}>
          Labels {state.showEdgeLabels ? 'ON' : 'OFF'}
        </ToolbarButton>

        <ToolbarButton
          className="min-w-[6.5rem]"
          onClick={handleFocusHighlights}
          title="Center the highlighted graph support"
        >
          Focus highlights
        </ToolbarButton>

        <label className="ml-1 flex min-w-[12.5rem] items-center gap-2 rounded-xl border border-base-content/8 bg-base-100/55 px-2.5 py-1.5">
          <span className="text-2xs font-medium uppercase tracking-[0.14em] text-base-content/42">Size</span>
          <select
            className="select select-xs h-8 min-h-8 flex-1 border-base-content/10 bg-base-100/75 text-xs"
            value={state.nodeSizeMetric}
            aria-label="Choose node sizing metric"
            onChange={(event) => handleSizeMetricChange(event.currentTarget.value as NodeSizeMetric)}
          >
            <option value="uniform">Uniform</option>
            <option value="degree">Degree</option>
            <option value="inDegree">In-degree</option>
            <option value="outDegree">Out-degree</option>
            <option value="pageRank">PageRank</option>
          </select>
        </label>
      </div>

      {/* Flexible space */}
      <div className="flex-1" />

      {/* Export */}
      <div className="flex shrink-0 items-center gap-1 rounded-2xl border border-base-content/10 bg-base-100/60 p-1">
        <ToolbarButton className="w-20" onClick={handleExportPNG} title="Export as PNG" aria-label="Export as PNG">
          <ArrowDownTrayIcon className="size-3.5 inline" aria-hidden="true" /> PNG
        </ToolbarButton>
        <ToolbarButton className="w-20" onClick={handleExportJSON} title="Export as JSON" aria-label="Export as JSON">
          <ArrowDownTrayIcon className="size-3.5 inline" aria-hidden="true" /> JSON
        </ToolbarButton>
      </div>
    </div>
  );
}
