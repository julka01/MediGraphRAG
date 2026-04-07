import { useCallback, useEffect } from 'react';
import type { AppAction, AppState, GraphData, GraphNode, GraphRelationship, ViewState } from '../types/app';
import { generateNodeTypeColors, generateRelationshipTypeColors } from '../utils/colors';
import { confidenceEdgeColor, getGraphTheme, normName } from '../utils/graph-helpers';

interface UseGraphOptions {
  containerRef: React.RefObject<HTMLDivElement | null>;
  appState: AppState;
  dispatch: React.Dispatch<AppAction>;
  networkRef: React.MutableRefObject<vis.Network | null>;
  idCounterRef: React.MutableRefObject<number>;
  initialViewRef: React.MutableRefObject<ViewState | null>;
  onNodeClick?: (node: Record<string, unknown> | null) => void;
}

export function useGraph({
  containerRef,
  appState,
  dispatch,
  networkRef,
  idCounterRef,
  initialViewRef,
  onNodeClick,
}: UseGraphOptions) {
  const { graphData, highlightedNodes, currentFilters } = appState;

  const initializeGraph = useCallback(
    (data: GraphData) => {
      if (!data || (!data.nodes && !data.relationships)) return;
      if (!containerRef.current) return;

      const gTheme = getGraphTheme();

      const nodeDegreeMap = new Map<string | number, number>();
      (data.relationships || []).forEach((rel: GraphRelationship) => {
        const f = rel.from || rel.start || rel.source;
        const t = rel.to || rel.end || rel.target;
        if (f != null) nodeDegreeMap.set(f, (nodeDegreeMap.get(f) || 0) + 1);
        if (t != null) nodeDegreeMap.set(t, (nodeDegreeMap.get(t) || 0) + 1);
      });

      if (networkRef.current) {
        try {
          networkRef.current.destroy();
        } catch (e) {
          console.warn('Error destroying network:', e);
        }
        networkRef.current = null;
      }

      idCounterRef.current = 0;

      const nodeTypes = new Set<string>();
      const relationshipTypes = new Set<string>();
      const getNodeType = (node: GraphNode): string => {
        const labels = node.labels ?? [];
        // Prefer specific type over generic __Entity__ / Chunk
        const specific = labels.find((l) => l !== '__Entity__' && l !== 'Chunk');
        return specific ?? labels[0] ?? 'Unknown';
      };
      (data.nodes || []).forEach((node: GraphNode) => {
        nodeTypes.add(getNodeType(node));
      });
      (data.relationships || []).forEach((rel: GraphRelationship) => {
        relationshipTypes.add(rel.type || 'Unknown');
      });

      const ntColors = generateNodeTypeColors(Array.from(nodeTypes));
      const rtColors = generateRelationshipTypeColors(Array.from(relationshipTypes));
      dispatch({ type: 'SET_NODE_TYPE_COLORS', colors: ntColors });
      dispatch({ type: 'SET_RELATIONSHIP_TYPE_COLORS', colors: rtColors });

      const normHighlighted = new Set([...highlightedNodes].map(normName));
      const referencedOriginalIds = new Set<string | number>();
      if (normHighlighted.size > 0) {
        (data.nodes || []).forEach((node: GraphNode) => {
          const p = node.properties || {};
          const candidates = [p.name as string, p.id as string, p.title as string].filter(Boolean);
          if (candidates.some((c) => normHighlighted.has(normName(c)))) {
            referencedOriginalIds.add(node.id);
          }
        });
      }
      const isHighlightMode = referencedOriginalIds.size > 0;

      const nodeIdMap = new Map<string | number, number>();

      const processedNodes = (data.nodes || [])
        .filter((node: GraphNode) => {
          if (currentFilters.nodeTypes.size > 0) {
            return currentFilters.nodeTypes.has(getNodeType(node));
          } else if (currentFilters.relationshipTypes.size > 0) {
            return data.relationships.some(
              (rel: GraphRelationship) =>
                currentFilters.relationshipTypes.has(rel.type || 'Unknown') &&
                ((rel.from || rel.start || rel.source) === node.id || (rel.to || rel.end || rel.target) === node.id),
            );
          }
          return true;
        })
        .map((node: GraphNode) => {
          const originalId = node.id;
          let newId: number;
          if (nodeIdMap.has(originalId)) {
            // Map.get is guaranteed to return a value here since we just checked .has()
            newId = nodeIdMap.get(originalId) ?? idCounterRef.current++;
          } else {
            newId = idCounterRef.current++;
            nodeIdMap.set(originalId, newId);
          }

          const nodeType = getNodeType(node);
          const nodeColor = ntColors[nodeType] || '#428bca';

          let displayLabel = (node.properties?.name as string) || getNodeType(node) || String(originalId);
          if (displayLabel.length > 20) displayLabel = `${displayLabel.substring(0, 17)}...`;

          const isReferenced = referencedOriginalIds.has(originalId);
          const degree = nodeDegreeMap.get(originalId) || 0;
          const MIN_WIDTH = 60;
          const MAX_WIDTH = 180;
          const maxDegree = Math.max(...Array.from(nodeDegreeMap.values()), 1);
          const nodeWidth = MIN_WIDTH + Math.round((degree / maxDegree) * (MAX_WIDTH - MIN_WIDTH));

          let finalColor: {
            background: string;
            border: string;
            highlight: { background: string; border: string };
            hover: { background: string; border: string };
          };
          let borderWidth: number, opacity: number;
          if (!isHighlightMode) {
            finalColor = {
              background: nodeColor,
              border: 'rgba(255,255,255,0.25)',
              highlight: { background: nodeColor, border: '#ffd700' },
              hover: { background: nodeColor, border: 'rgba(255,255,255,0.6)' },
            };
            borderWidth = 2;
            opacity = 1;
          } else if (isReferenced) {
            finalColor = {
              background: nodeColor,
              border: '#ffd700',
              highlight: { background: nodeColor, border: '#ffaa00' },
              hover: { background: nodeColor, border: '#ffd700' },
            };
            borderWidth = 4;
            opacity = 1;
          } else {
            finalColor = {
              background: gTheme.dimmedNodeBg,
              border: gTheme.dimmedNodeBdr,
              highlight: { background: gTheme.dimmedNodeBdr, border: gTheme.dimmedNodeBdr },
              hover: { background: gTheme.dimmedNodeBg, border: gTheme.dimmedNodeBdr },
            };
            borderWidth = 1;
            opacity = 0.35;
          }

          return {
            id: newId,
            label: displayLabel,
            originalId,
            labels: node.labels,
            properties: node.properties,
            title: `${nodeType}: ${displayLabel}\n\nLabels: ${node.labels?.join(', ') || 'Unknown'}\nID: ${originalId}\n\n${
              node.properties
                ? Object.entries(node.properties)
                    .map(([k, v]) => `${k}: ${v}`)
                    .join('\n')
                : 'No properties'
            }`,
            color: finalColor,
            opacity,
            margin: 5,
            widthConstraint: { minimum: nodeWidth, maximum: nodeWidth },
            borderWidth,
            font: {
              size: isReferenced ? 13 : 11,
              face: 'Helvetica, Arial, sans-serif',
              color: isHighlightMode && !isReferenced ? gTheme.nodeTextDimmed : gTheme.nodeText,
              strokeWidth: 0,
              align: 'center',
              vadjust: 0,
            },
            shape: 'circle',
            shadow: false,
            _baseWidth: nodeWidth,
            _baseColor: nodeColor,
          };
        });

      const processedEdges = (data.relationships || [])
        .filter((rel: GraphRelationship) => {
          if (currentFilters.relationshipTypes.size > 0) {
            return currentFilters.relationshipTypes.has(rel.type || 'Unknown');
          }
          return true;
        })
        .map((rel: GraphRelationship) => {
          const fromOrigId = rel.from || rel.start || rel.source;
          const toOrigId = rel.to || rel.end || rel.target;
          if (fromOrigId == null || toOrigId == null) return null;
          const fromId = nodeIdMap.get(fromOrigId);
          const toId = nodeIdMap.get(toOrigId);
          if (fromId === undefined || toId === undefined) return null;

          const relType = rel.type || 'Unknown';
          const edgeColor = rtColors[relType] || '#888888';
          const confidence = (rel.properties?.confidence as number) ?? null;
          const confColor = confidenceEdgeColor(confidence);
          const sourceReferenced = referencedOriginalIds.has(fromOrigId);
          const targetReferenced = referencedOriginalIds.has(toOrigId);
          const isHighlightedEdge = sourceReferenced && targetReferenced;
          const isDimmed = isHighlightMode && !isHighlightedEdge;
          const normalColor = confColor || edgeColor;
          const confWidth = confidence != null ? 0.8 + confidence * 2.2 : 1;
          const confLabel = confidence != null ? `\nConfidence: ${(confidence * 100).toFixed(0)}%` : '';

          return {
            id: idCounterRef.current++,
            from: fromId,
            to: toId,
            label: relType,
            originalId: rel.id,
            arrows: { to: { enabled: true, scaleFactor: 0.8, type: 'arrow' } },
            color: {
              color: isDimmed ? gTheme.dimmedEdge : isHighlightedEdge ? '#ffd700' : normalColor,
              highlight: '#ffd700',
              hover: '#ffd700',
              opacity: isDimmed ? 0.25 : 1.0,
            },
            font: {
              size: 11,
              face: 'Helvetica, Arial, sans-serif',
              color: isDimmed ? gTheme.nodeTextDimmed : isHighlightedEdge ? '#ffd700' : gTheme.edgeText,
              background: gTheme.edgeLabelBg,
              strokeWidth: 0,
              align: 'top',
              vadjust: 15,
            },
            title: `${relType}\nFrom: ${fromOrigId}\nTo: ${toOrigId}${confLabel}`,
            width: isHighlightedEdge ? 3 : confWidth,
            _baseColor: normalColor,
            _baseWidth: confWidth,
            smooth: { enabled: true, type: 'straightCross', forceDirection: false },
            shadow: false,
          };
        })
        .filter(Boolean);

      if (processedNodes.length === 0) return;

      const nodes = new vis.DataSet(processedNodes);
      const edges = new vis.DataSet(processedEdges.filter((e): e is NonNullable<typeof e> => e != null));

      const graphSize = processedNodes.length;
      const isLargeGraph = graphSize > 500;
      const isVeryLargeGraph = graphSize > 1000;

      let physicsSettings: Record<string, unknown>;
      if (isVeryLargeGraph) {
        physicsSettings = { enabled: false, stabilization: false };
      } else if (isLargeGraph) {
        physicsSettings = {
          enabled: true,
          stabilization: { enabled: true, iterations: 500, updateInterval: 25 },
          barnesHut: {
            gravitationalConstant: -2000,
            centralGravity: 0.3,
            springLength: 100,
            springConstant: 0.04,
            damping: 0.25,
            avoidOverlap: 0.02,
          },
          solver: 'barnesHut',
          minVelocity: 0.5,
          maxVelocity: 50,
          timestep: 0.4,
        };
      } else {
        physicsSettings = {
          enabled: true,
          stabilization: { enabled: true, iterations: 500, updateInterval: 25 },
          barnesHut: {
            gravitationalConstant: -5000,
            centralGravity: 0.3,
            springLength: 150,
            springConstant: 0.04,
            damping: 0.09,
            avoidOverlap: 0.1,
          },
          solver: 'barnesHut',
          minVelocity: 0.75,
          maxVelocity: 100,
          timestep: 0.35,
        };
      }

      const options = {
        nodes: {
          shape: 'circle',
          size: isLargeGraph ? 20 : 30,
          font: {
            size: isLargeGraph ? 0 : 12,
            face: 'Helvetica, Arial, sans-serif',
            color: gTheme.nodeText,
            strokeWidth: 0,
            align: 'center',
            vadjust: 0,
          },
          borderWidth: 1,
          borderWidthSelected: 3,
          shadow: false,
        },
        edges: {
          width: 1,
          arrows: { to: { enabled: true, scaleFactor: 0.8, type: 'arrow' } },
          font: {
            size: isLargeGraph ? 0 : 11,
            face: 'Helvetica, Arial, sans-serif',
            color: gTheme.edgeText,
            background: gTheme.edgeLabelBg,
            strokeWidth: 0,
            align: 'top',
            vadjust: 15,
          },
          color: { inherit: 'from' },
          smooth: { enabled: !isLargeGraph, type: 'straightCross', forceDirection: false },
          shadow: false,
        },
        physics: physicsSettings,
        interaction: {
          hover: !isLargeGraph,
          tooltipDelay: 200,
          zoomView: true,
          dragView: true,
          dragNodes: !isVeryLargeGraph,
          multiselect: false,
          navigationButtons: false,
          keyboard: true,
          selectConnectedEdges: false,
        },
        layout: { improvedLayout: !isLargeGraph, hierarchical: false },
        configure: { enabled: false },
      };

      try {
        networkRef.current = new vis.Network(containerRef.current, { nodes, edges }, options);

        networkRef.current.on('stabilizationIterationsDone', () => {
          networkRef.current?.setOptions({ physics: { enabled: true, stabilization: false } });
          setTimeout(() => {
            try {
              if (networkRef.current) {
                initialViewRef.current = {
                  scale: networkRef.current.getScale(),
                  position: networkRef.current.getViewPosition(),
                };
              }
            } catch {
              /* ignore */
            }
          }, 1000);
        });

        networkRef.current.on('click', (params: Record<string, unknown>) => {
          const nodeIds = params.nodes as number[];
          if (!nodeIds || nodeIds.length === 0) {
            onNodeClick?.(null);
            return;
          }
          const nodeId = nodeIds[0];
          const node = networkRef.current?.body.data.nodes.get(nodeId);
          if (node) onNodeClick?.(node as Record<string, unknown>);
        });
      } catch (error) {
        console.error('Error creating network:', error);
      }
    },
    [containerRef, networkRef, idCounterRef, initialViewRef, dispatch, onNodeClick, highlightedNodes, currentFilters],
  );

  useEffect(() => {
    if (graphData) {
      initializeGraph(graphData);
    }
  }, [graphData, initializeGraph]);

  return { initializeGraph };
}
