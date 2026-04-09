import { useState, useRef, useCallback, useEffect } from 'react';
import { SunIcon, MoonIcon } from '@heroicons/react/24/outline';
import { useApp } from '../../context/AppContext';
import { useTheme } from '../../context/ThemeContext';
import { useGraph } from '../../hooks/useGraph';
import { NodeDetailPanel } from './NodeDetailPanel';
import { ProgressPanel } from '../kg/ProgressPanel';
import { PanelToggleIcon } from './PanelToggleIcons';

interface GraphContainerProps {
  progressActive: boolean;
  onProgressClose: () => void;
}

interface NodeEdge {
  from: string;
  to: string;
  label: string;
  toLabel: string;
  fromLabel: string;
}

export function GraphContainer({ progressActive, onProgressClose }: GraphContainerProps) {
  const { state, dispatch, networkRef, idCounterRef, initialViewRef } = useApp();
  const { theme, toggleTheme } = useTheme();
  const [selectedNode, setSelectedNode] = useState<Record<string, unknown> | null>(null);
  const [selectedNodeEdges, setSelectedNodeEdges] = useState<NodeEdge[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);

  const getNodeEdges = useCallback((nodeId: number): NodeEdge[] => {
    const net = networkRef.current;
    if (!net) return [];
    const edgeDS = (net.body as { data: { edges: { get: () => Array<Record<string, unknown>> } } }).data.edges;
    const nodeDS = (net.body as { data: { nodes: { get: (id: number) => Record<string, unknown> | null } } }).data.nodes;
    const allEdges = edgeDS.get();
    return allEdges
      .filter((e) => e.from === nodeId || e.to === nodeId)
      .map((e) => {
        const targetId = e.from === nodeId ? e.to : e.from;
        const targetNode = nodeDS.get(targetId as number);
        return {
          from: String(e.from),
          to: String(e.to),
          label: String(e.label ?? e.title ?? ''),
          toLabel: targetNode ? String(targetNode.label ?? '') : '',
          fromLabel: '',
        };
      });
  }, [networkRef]);

  const handleNodeClick = useCallback((node: Record<string, unknown> | null) => {
    setSelectedNode(node);
    if (node) {
      const nodeId = (node.id ?? node.originalId) as number;
      setSelectedNodeEdges(getNodeEdges(nodeId));
    } else {
      setSelectedNodeEdges([]);
    }
  }, [getNodeEdges]);

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
    function handleEscape(e: KeyboardEvent) {
      if (e.key === 'Escape') setSelectedNode(null);
    }
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, []);

  const hasGraph = state.graphData && state.graphData.nodes.length > 0;
  const { leftCollapsed, bottomCollapsed, rightCollapsed } = state.panels;

  const fallbackColor =
    getComputedStyle(document.documentElement).getPropertyValue('--color-primary').trim() || '#428bca';
  const nodeColor = selectedNode
    ? state.nodeTypeColors[(selectedNode.labels as string[] | undefined)?.[0] || 'Unknown'] || fallbackColor
    : fallbackColor;

  return (
    <div className="relative flex flex-col h-full bg-base-100 min-w-[300px]">
      {/* Graph canvas */}
      <div ref={containerRef} className="flex-1 min-h-0">
        {!hasGraph && !progressActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-base-content/40">
            <span className="text-5xl mb-2">&#9672;</span>
            <p className="text-sm">Load or create a knowledge graph to visualize</p>
          </div>
        )}
      </div>

      {/* Theme toggle — top left */}
      <div className="absolute top-2 left-2 z-10">
        <button
          type="button"
          onClick={toggleTheme}
          className="btn btn-ghost btn-sm btn-square"
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
        >
          {theme === 'dark' ? <SunIcon className="size-4" /> : <MoonIcon className="size-4" />}
        </button>
      </div>

      {/* Panel toggles — top right */}
      <div className="absolute top-2 right-2 z-10 flex gap-0.5 bg-base-200/80 backdrop-blur rounded-lg p-0.5">
        <PanelToggleIcon
          panel="left"
          isOpen={!leftCollapsed}
          onClick={() => dispatch({ type: 'TOGGLE_LEFT_PANEL' })}
        />
        <PanelToggleIcon
          panel="bottom"
          isOpen={!bottomCollapsed}
          onClick={() => dispatch({ type: 'TOGGLE_BOTTOM_PANEL' })}
        />
        <PanelToggleIcon
          panel="right"
          isOpen={!rightCollapsed}
          onClick={() => dispatch({ type: 'TOGGLE_RIGHT_PANEL' })}
        />
      </div>

      {/* Node detail panel */}
      {selectedNode && (
        <NodeDetailPanel
          node={selectedNode}
          nodeColor={nodeColor}
          edges={selectedNodeEdges}
          onClose={() => { setSelectedNode(null); setSelectedNodeEdges([]); }}
        />
      )}

      {/* Progress overlay */}
      <ProgressPanel active={progressActive} onClose={onProgressClose} />
    </div>
  );
}
