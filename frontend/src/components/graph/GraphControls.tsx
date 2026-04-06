import { useApp } from '../../context/AppContext';
import { showError, showSuccess } from '../ui/Notifications';

export function GraphControls() {
  const { state, dispatch, networkRef, initialViewRef } = useApp();
  const network = networkRef.current;

  const handleZoomIn = () => {
    if (!network) return;
    network.moveTo({ scale: network.getScale() * 1.2, animation: true });
  };

  const handleZoomOut = () => {
    if (!network) return;
    network.moveTo({ scale: network.getScale() * 0.8, animation: true });
  };

  const handleResetZoom = () => {
    if (!network) return;
    dispatch({ type: 'CLEAR_FILTERS' });
    dispatch({ type: 'CLEAR_HIGHLIGHTED_NODES' });
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

  const handlePhysicsToggle = () => {
    dispatch({ type: 'TOGGLE_PHYSICS' });
    if (network) {
      network.setOptions({ physics: { enabled: !state.physicsEnabled } });
    }
  };

  const handleEdgeLabelsToggle = () => {
    dispatch({ type: 'TOGGLE_EDGE_LABELS' });
  };

  const handleExportPNG = () => {
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

  const handleClearHighlights = () => {
    dispatch({ type: 'CLEAR_HIGHLIGHTED_NODES' });
  };

  return (
    <div className="flex items-center gap-1 flex-wrap">
      <div className="join">
        <button
          type="button"
          className="btn btn-ghost btn-xs join-item"
          onClick={handleZoomIn}
          title="Zoom in"
          aria-label="Zoom in"
        >
          +
        </button>
        <button
          type="button"
          className="btn btn-ghost btn-xs join-item"
          onClick={handleZoomOut}
          title="Zoom out"
          aria-label="Zoom out"
        >
          −
        </button>
        <button
          type="button"
          className="btn btn-ghost btn-xs join-item"
          onClick={handleResetZoom}
          title="Reset view"
          aria-label="Reset view"
        >
          ↺
        </button>
      </div>

      <label className="flex items-center gap-1 cursor-pointer text-xs">
        <input
          type="checkbox"
          className="toggle toggle-xs"
          checked={state.physicsEnabled}
          onChange={handlePhysicsToggle}
        />
        Physics
      </label>

      <label className="flex items-center gap-1 cursor-pointer text-xs">
        <input
          type="checkbox"
          className="toggle toggle-xs"
          checked={state.showEdgeLabels}
          onChange={handleEdgeLabelsToggle}
        />
        Labels
      </label>

      <select
        className="select select-ghost select-xs"
        value={state.nodeSizeMetric}
        onChange={(e) => dispatch({ type: 'SET_NODE_SIZE_METRIC', metric: e.target.value })}
      >
        <option value="degree">Size: Degree</option>
        <option value="uniform">Size: Uniform</option>
      </select>

      <button
        type="button"
        className="btn btn-ghost btn-xs"
        onClick={handleExportPNG}
        title="Export as PNG"
        aria-label="Export as PNG"
      >
        ↓ PNG
      </button>
      <button
        type="button"
        className="btn btn-ghost btn-xs"
        onClick={handleExportJSON}
        title="Export as JSON"
        aria-label="Export as JSON"
      >
        ↓ JSON
      </button>

      {state.highlightedNodes.size > 0 && (
        <button
          type="button"
          className="btn btn-ghost btn-xs text-warning"
          onClick={handleClearHighlights}
          title="Clear highlights"
        >
          ✕ {state.highlightedNodes.size} highlighted
        </button>
      )}
    </div>
  );
}
