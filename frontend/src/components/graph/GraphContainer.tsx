import { MoonIcon, SunIcon } from '@heroicons/react/24/outline';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { useTheme } from '../../context/ThemeContext';
import { useGraph } from '../../hooks/useGraph';
import { ProgressPanel } from '../kg/ProgressPanel';
import { NodeDetailPanel } from './NodeDetailPanel';
import { PanelToggleIcon } from './PanelToggleIcons';
import { applySearchToNetwork } from './TopBar';

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

  const getNodeEdges = useCallback(
    (nodeId: number): NodeEdge[] => {
      const net = networkRef.current;
      if (!net) return [];
      const edgeDS = (net.body as { data: { edges: { get: () => Array<Record<string, unknown>> } } }).data.edges;
      const nodeDS = (net.body as { data: { nodes: { get: (id: number) => Record<string, unknown> | null } } }).data
        .nodes;
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
    },
    [networkRef],
  );

  const handleNodeClick = useCallback(
    (node: Record<string, unknown> | null) => {
      setSelectedNode(node);
      if (node) {
        const nodeId = (node.id ?? node.originalId) as number;
        setSelectedNodeEdges(getNodeEdges(nodeId));
      } else {
        setSelectedNodeEdges([]);
      }
    },
    [getNodeEdges],
  );

  const applySearch = useCallback((term: string) => applySearchToNetwork(networkRef, term), [networkRef]);

  useGraph({
    containerRef,
    appState: state,
    dispatch,
    networkRef,
    idCounterRef,
    initialViewRef,
    onNodeClick: handleNodeClick,
    applySearch,
  });

  useEffect(() => {
    function handleEscape(e: KeyboardEvent) {
      if (e.key === 'Escape') setSelectedNode(null);
    }
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, []);

  const hasGraph = state.graphData && state.graphData.nodes.length > 0;
  const { leftCollapsed, bottomCollapsed, rightCollapsed, topCollapsed } = state.panels;

  // biome-ignore lint/correctness/useExhaustiveDependencies: theme controls CSS variable value
  const fallbackColor = useMemo(
    () => getComputedStyle(document.documentElement).getPropertyValue('--color-primary').trim() || '#428bca',
    [theme],
  );
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

      {/* Panel toggles — top left, cross layout with theme toggle in center */}
      <div className="absolute top-2 left-2 z-10 grid grid-cols-3 grid-rows-3 bg-base-200/80 backdrop-blur rounded-lg p-0.5">
        {/* Row 1: top toggle centered */}
        <div />
        <PanelToggleIcon panel="top" isOpen={!topCollapsed} onClick={() => dispatch({ type: 'TOGGLE_TOP_PANEL' })} />
        <div />

        {/* Row 2: left, theme toggle center, right */}
        <PanelToggleIcon panel="left" isOpen={!leftCollapsed} onClick={() => dispatch({ type: 'TOGGLE_LEFT_PANEL' })} />
        <button
          type="button"
          onClick={toggleTheme}
          className="p-1 rounded-md transition-colors text-base-content/50 hover:text-base-content/80"
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
        >
          {theme === 'dark' ? <SunIcon className="size-[22px]" /> : <MoonIcon className="size-[22px]" />}
        </button>
        <PanelToggleIcon
          panel="right"
          isOpen={!rightCollapsed}
          onClick={() => dispatch({ type: 'TOGGLE_RIGHT_PANEL' })}
        />

        {/* Row 3: bottom toggle centered */}
        <div />
        <PanelToggleIcon
          panel="bottom"
          isOpen={!bottomCollapsed}
          onClick={() => dispatch({ type: 'TOGGLE_BOTTOM_PANEL' })}
        />
        <div />
      </div>

      {/* Node detail panel */}
      {selectedNode && (
        <NodeDetailPanel
          node={selectedNode}
          nodeColor={nodeColor}
          edges={selectedNodeEdges}
          onClose={() => {
            setSelectedNode(null);
            setSelectedNodeEdges([]);
          }}
        />
      )}

      {/* Progress overlay */}
      <ProgressPanel active={progressActive} onClose={onProgressClose} />
    </div>
  );
}
