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
          const fromNode = nodeDS.get(e.from as number);
          const toNode = nodeDS.get(e.to as number);
          return {
            from: String(e.from),
            to: String(e.to),
            label: String(e.label ?? e.title ?? ''),
            toLabel: toNode ? String(toNode.label ?? '') : '',
            fromLabel: fromNode ? String(fromNode.label ?? '') : '',
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
    <div className="relative flex h-full min-w-[300px] flex-col bg-base-100">
      {/* Graph canvas */}
      <div ref={containerRef} className="flex-1 min-h-0">
        {!hasGraph && !progressActive && (
          <div className="absolute inset-0 flex items-center justify-center p-6">
            <div className="panel-glass max-w-md rounded-3xl px-6 py-7 text-center">
              <div className="mx-auto mb-4 flex size-14 items-center justify-center rounded-2xl bg-primary/10 text-xl text-primary">
                ◇
              </div>
              <p className="text-[0.65rem] font-medium uppercase tracking-[0.22em] text-base-content/45">
                Graph canvas
              </p>
              <h3 className="mt-2 text-lg font-semibold text-base-content">Load or create a knowledge graph</h3>
              <p className="mt-2 text-sm leading-6 text-base-content/58">
                Build a graph from the workspace panel, then search, inspect, and trace evidence here.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Panel toggles — top left, cross layout with theme toggle in center */}
      <div className="panel-glass absolute left-3 top-3 z-10 grid grid-cols-3 grid-rows-3 rounded-2xl p-1">
        {/* Row 1: top toggle centered */}
        <div />
        <PanelToggleIcon panel="top" isOpen={!topCollapsed} onClick={() => dispatch({ type: 'TOGGLE_TOP_PANEL' })} />
        <div />

        {/* Row 2: left, theme toggle center, right */}
        <PanelToggleIcon panel="left" isOpen={!leftCollapsed} onClick={() => dispatch({ type: 'TOGGLE_LEFT_PANEL' })} />
        <label className="swap swap-rotate p-1 text-base-content/50 hover:text-base-content/80">
          <input
            type="checkbox"
            className="hidden"
            value="light"
            checked={theme === 'light'}
            onChange={toggleTheme}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          />
          <SunIcon className="swap-on size-[22px]" />
          <MoonIcon className="swap-off size-[22px]" />
        </label>
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
