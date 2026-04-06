import { useEffect, useRef, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { useGraph } from '../../hooks/useGraph';
import { ProgressPanel } from '../kg/ProgressPanel';
import { Panel } from '../ui/Panel';
import { GraphControls } from './GraphControls';
import { GraphFilters } from './GraphFilters';
import { GraphLegend, MiniLegend } from './GraphLegend';
import { GraphSearch } from './GraphSearch';
import { NodeDetailPanel } from './NodeDetailPanel';

interface GraphContainerProps {
  progressActive: boolean;
  onProgressClose: () => void;
}

export function GraphContainer({ progressActive, onProgressClose }: GraphContainerProps) {
  const { state, dispatch, networkRef, idCounterRef, initialViewRef } = useApp();
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedNode, setSelectedNode] = useState<Record<string, unknown> | null>(null);

  const handleNodeClick = (node: Record<string, unknown> | null) => {
    setSelectedNode(node);
  };

  useGraph({
    containerRef,
    appState: state,
    dispatch,
    networkRef,
    idCounterRef,
    initialViewRef,
    onNodeClick: handleNodeClick,
  });

  useEffect(() => {
    if (!selectedNode) return;
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSelectedNode(null);
    };
    document.addEventListener('keydown', handleEsc);
    return () => document.removeEventListener('keydown', handleEsc);
  }, [selectedNode]);

  const hasGraph = !!state.graphData;
  const nodeColor = selectedNode
    ? state.nodeTypeColors[(selectedNode.labels as string[] | undefined)?.[0] || 'Unknown'] || '#428bca'
    : '#428bca';

  return (
    <Panel>
      <Panel.Header
        title="Knowledge Graph"
        badge={
          state.currentKGName ? <span className="badge badge-sm badge-primary">{state.currentKGName}</span> : undefined
        }
      >
        <button
          type="button"
          className="btn btn-ghost btn-xs"
          onClick={() => dispatch({ type: 'TOGGLE_KG_EXPANDED' })}
          title={state.kgExpanded ? 'Collapse graph view' : 'Expand graph view'}
        >
          {state.kgExpanded ? '\u229F' : '\u229E'}
        </button>
      </Panel.Header>

      {state.highlightedNodes.size > 0 && (
        <div className="px-2 mb-1">
          <span className="badge badge-warning badge-sm gap-1">
            {state.highlightedNodes.size} entities highlighted
            <button
              type="button"
              className="btn btn-ghost btn-xs px-0"
              onClick={() => dispatch({ type: 'CLEAR_HIGHLIGHTED_NODES' })}
            >
              &#x2715;
            </button>
          </span>
        </div>
      )}

      <div className="px-2">
        <MiniLegend />
      </div>

      {hasGraph && (
        <div className="flex items-center gap-2 px-2 py-1 flex-wrap">
          <GraphSearch />
          <GraphFilters />
        </div>
      )}

      <Panel.Body scrollable={false}>
        <div className="relative w-full h-full">
          <div ref={containerRef} className="w-full h-full" />

          {!hasGraph && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-base-content/40">
              <div className="text-4xl mb-2">&#x2B21;</div>
              <p className="text-sm">No graph loaded yet. Upload a document or load from Neo4j to get started.</p>
            </div>
          )}

          {selectedNode && (
            <NodeDetailPanel node={selectedNode} nodeColor={nodeColor} onClose={() => setSelectedNode(null)} />
          )}

          <GraphLegend />
          <ProgressPanel active={progressActive} onClose={onProgressClose} />
        </div>
      </Panel.Body>

      <Panel.Footer>
        <GraphControls />
      </Panel.Footer>
    </Panel>
  );
}
