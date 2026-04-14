import { useCallback, useEffect, useRef } from 'react';
import type { AppAction, AppState, GraphData, GraphNode, GraphRelationship, ViewState } from '../types/app';
import { generateNodeTypeColors, generateRelationshipTypeColors } from '../utils/colors';
import { getGraphTheme, normName } from '../utils/graph-helpers';

interface UseGraphOptions {
  containerRef: React.RefObject<HTMLDivElement | null>;
  appState: AppState;
  dispatch: React.Dispatch<AppAction>;
  networkRef: React.MutableRefObject<vis.Network | null>;
  idCounterRef: React.MutableRefObject<number>;
  initialViewRef: React.MutableRefObject<ViewState | null>;
  onNodeClick?: (node: Record<string, unknown> | null) => void;
  applySearch?: (term: string) => void;
}

function parseColor(color: string) {
  const value = color.trim();
  const hex = value.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (hex) {
    const raw = hex[1];
    const expanded =
      raw.length === 3
        ? raw
            .split('')
            .map((char) => char + char)
            .join('')
        : raw;
    return {
      r: Number.parseInt(expanded.slice(0, 2), 16),
      g: Number.parseInt(expanded.slice(2, 4), 16),
      b: Number.parseInt(expanded.slice(4, 6), 16),
    };
  }

  const rgb = value.match(/^rgba?\((\d+),\s*(\d+),\s*(\d+)/i);
  if (rgb) {
    return {
      r: Number.parseInt(rgb[1], 10),
      g: Number.parseInt(rgb[2], 10),
      b: Number.parseInt(rgb[3], 10),
    };
  }

  return null;
}

function blendColors(from: string, to: string, weight: number) {
  const source = parseColor(from);
  const target = parseColor(to);
  if (!source || !target) return to;

  const clamped = Math.min(1, Math.max(0, weight));
  const mix = (a: number, b: number) => Math.round(a + (b - a) * clamped);
  return `rgb(${mix(source.r, target.r)}, ${mix(source.g, target.g)}, ${mix(source.b, target.b)})`;
}

function withAlpha(color: string, alpha: number) {
  const parsed = parseColor(color);
  if (!parsed) return color;
  const clamped = Math.min(1, Math.max(0, alpha));
  return `rgba(${parsed.r}, ${parsed.g}, ${parsed.b}, ${clamped})`;
}

function scaleNodeWidth(baseWidth: number, value: number, maxValue: number) {
  if (maxValue <= 0 || value <= 0) return baseWidth;
  const normalized = Math.min(1, value / maxValue);
  return Math.round(baseWidth * (1 + 2.6 * Math.sqrt(normalized)));
}

function computePageRank(nodeIds: Array<string | number>, relationships: GraphRelationship[], damping = 0.85, iterations = 24) {
  const pageRank = new Map<string | number, number>();
  const outbound = new Map<string | number, Array<string | number>>();
  const nodeSet = new Set(nodeIds);
  const count = nodeIds.length;
  if (count === 0) return pageRank;

  const initialScore = 1 / count;
  nodeIds.forEach((nodeId) => {
    pageRank.set(nodeId, initialScore);
    outbound.set(nodeId, []);
  });

  relationships.forEach((rel) => {
    const fromId = rel.from || rel.start || rel.source;
    const toId = rel.to || rel.end || rel.target;
    if (fromId == null || toId == null) return;
    if (!nodeSet.has(fromId) || !nodeSet.has(toId)) return;
    outbound.get(fromId)?.push(toId);
  });

  for (let step = 0; step < iterations; step += 1) {
    const next = new Map<string | number, number>();
    nodeIds.forEach((nodeId) => {
      next.set(nodeId, (1 - damping) / count);
    });

    let danglingMass = 0;
    nodeIds.forEach((nodeId) => {
      const score = pageRank.get(nodeId) ?? 0;
      const targets = outbound.get(nodeId) ?? [];
      if (targets.length === 0) {
        danglingMass += score;
        return;
      }
      const share = (damping * score) / targets.length;
      targets.forEach((targetId) => {
        next.set(targetId, (next.get(targetId) ?? 0) + share);
      });
    });

    const danglingShare = (damping * danglingMass) / count;
    nodeIds.forEach((nodeId) => {
      next.set(nodeId, (next.get(nodeId) ?? 0) + danglingShare);
    });

    pageRank.clear();
    next.forEach((value, key) => pageRank.set(key, value));
  }

  return pageRank;
}

function getPhysicsSettings(isLargeGraph: boolean, spacing: number) {
  const clamped = Math.min(1, Math.max(0, spacing));
  const spacingFactor = 0.72 + clamped * 1.28;

  return {
    enabled: true,
    stabilization: {
      enabled: true,
      iterations: Math.round((isLargeGraph ? 360 : 280) + clamped * (isLargeGraph ? 220 : 180)),
      updateInterval: 25,
    },
    barnesHut: {
      gravitationalConstant: Math.round((isLargeGraph ? -2300 : -1550) * (0.84 + clamped * 1.06)),
      centralGravity: Math.max(0.08, (isLargeGraph ? 0.34 : 0.42) - clamped * 0.22),
      springLength: Math.round((isLargeGraph ? 94 : 118) * spacingFactor),
      springConstant: Math.max(0.018, (isLargeGraph ? 0.06 : 0.072) - clamped * 0.036),
      damping: Math.max(0.16, (isLargeGraph ? 0.5 : 0.56) - clamped * 0.22),
      avoidOverlap: (isLargeGraph ? 0.12 : 0.18) + clamped * (isLargeGraph ? 0.12 : 0.18),
    },
    solver: 'barnesHut',
    minVelocity: Math.max(0.45, 1.15 - clamped * 0.55),
    maxVelocity: Math.round((isLargeGraph ? 22 : 28) + clamped * (isLargeGraph ? 26 : 36)),
    timestep: 0.32 + clamped * 0.08,
  };
}

export function useGraph({
  containerRef,
  appState,
  dispatch,
  networkRef,
  idCounterRef,
  initialViewRef,
  onNodeClick,
  applySearch,
}: UseGraphOptions) {
  const { graphData, highlightedNodes, currentFilters, searchTerm, physicsEnabled, layoutSpacing, showEdgeLabels, nodeSizeMetric } =
    appState;

  // Refs for toggle state so initializeGraph reads latest values without re-creating
  const physicsEnabledRef = useRef(physicsEnabled);
  const spacingRef = useRef(layoutSpacing);
  const labelsRef = useRef(showEdgeLabels);
  const sizeMetricRef = useRef(nodeSizeMetric);
  physicsEnabledRef.current = physicsEnabled;
  spacingRef.current = layoutSpacing;
  labelsRef.current = showEdgeLabels;
  sizeMetricRef.current = nodeSizeMetric;

  const initializeGraph = useCallback(
    (data: GraphData) => {
      if (!data || (!data.nodes && !data.relationships)) return;
      if (!containerRef.current) return;

      const gTheme = getGraphTheme();
      const allNodes = data.nodes || [];
      const allRelationships = data.relationships || [];

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
        allNodes.forEach((node: GraphNode) => {
          const p = node.properties || {};
          const candidates = [p.name as string, p.id as string, p.title as string].filter(Boolean);
          if (candidates.some((c) => normHighlighted.has(normName(c)))) {
            referencedOriginalIds.add(node.id);
          }
        });
      }
      const isHighlightMode = referencedOriginalIds.size > 0;

      const nodeIdMap = new Map<string | number, number>();
      const BASE_WIDTH = 60;
      const filteredNodes = allNodes
        .filter((node: GraphNode) => {
          const nodeFilter = currentFilters.nodeTypes;
          const relFilter = currentFilters.relationshipTypes;

          // null = uninitialized, show all; empty Set = user cleared all, show none
          if (nodeFilter === null && relFilter === null) return true;

          // If both are explicit empty sets, user deselected everything
          if (nodeFilter !== null && nodeFilter.size === 0 && relFilter !== null && relFilter.size === 0) return false;

          if (nodeFilter !== null && nodeFilter.size > 0) {
            return nodeFilter.has(getNodeType(node));
          }
          if (relFilter !== null && relFilter.size > 0) {
            return allRelationships.some(
              (rel: GraphRelationship) =>
                relFilter.has(rel.type || 'Unknown') &&
                ((rel.from || rel.start || rel.source) === node.id || (rel.to || rel.end || rel.target) === node.id),
            );
          }
          return true;
        });

      const visibleOriginalIds = new Set(filteredNodes.map((node) => node.id));
      const filteredRelationships = allRelationships.filter((rel: GraphRelationship) => {
        const relFilter = currentFilters.relationshipTypes;
        if (relFilter !== null && relFilter.size === 0) return false;
        if (relFilter !== null && relFilter.size > 0 && !relFilter.has(rel.type || 'Unknown')) return false;

        const fromOrigId = rel.from || rel.start || rel.source;
        const toOrigId = rel.to || rel.end || rel.target;
        if (fromOrigId == null || toOrigId == null) return false;
        return visibleOriginalIds.has(fromOrigId) && visibleOriginalIds.has(toOrigId);
      });

      const totalDegreeMap = new Map<string | number, number>();
      const inDegreeMap = new Map<string | number, number>();
      const outDegreeMap = new Map<string | number, number>();
      filteredRelationships.forEach((rel: GraphRelationship) => {
        const fromOrigId = rel.from || rel.start || rel.source;
        const toOrigId = rel.to || rel.end || rel.target;
        if (fromOrigId == null || toOrigId == null) return;
        outDegreeMap.set(fromOrigId, (outDegreeMap.get(fromOrigId) || 0) + 1);
        inDegreeMap.set(toOrigId, (inDegreeMap.get(toOrigId) || 0) + 1);
        totalDegreeMap.set(fromOrigId, (totalDegreeMap.get(fromOrigId) || 0) + 1);
        totalDegreeMap.set(toOrigId, (totalDegreeMap.get(toOrigId) || 0) + 1);
      });

      const pageRankMap = computePageRank(Array.from(visibleOriginalIds), filteredRelationships);
      const maxDegree = Math.max(...Array.from(totalDegreeMap.values()), 1);
      const maxInDegree = Math.max(...Array.from(inDegreeMap.values()), 1);
      const maxOutDegree = Math.max(...Array.from(outDegreeMap.values()), 1);
      const maxPageRank = Math.max(...Array.from(pageRankMap.values()), 1e-6);

      const processedNodes = filteredNodes
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
          const nodeColor = ntColors[nodeType] || gTheme.highlight;

          let displayLabel = (node.properties?.name as string) || getNodeType(node) || String(originalId);
          if (displayLabel.length > 20) displayLabel = `${displayLabel.substring(0, 17)}\u2026`;

          const isReferenced = referencedOriginalIds.has(originalId);
          const degree = totalDegreeMap.get(originalId) || 0;
          const inDegree = inDegreeMap.get(originalId) || 0;
          const outDegree = outDegreeMap.get(originalId) || 0;
          const pageRank = pageRankMap.get(originalId) || 0;
          const pageRankNorm = maxPageRank > 0 ? Math.min(1, pageRank / maxPageRank) : 0;
          const pageRankHeatEnabled = sizeMetricRef.current === 'pageRank';
          const metricWidths = {
            uniform: BASE_WIDTH,
            degree: scaleNodeWidth(BASE_WIDTH, degree, maxDegree),
            inDegree: scaleNodeWidth(BASE_WIDTH, inDegree, maxInDegree),
            outDegree: scaleNodeWidth(BASE_WIDTH, outDegree, maxOutDegree),
            pageRank: scaleNodeWidth(BASE_WIDTH, pageRank, maxPageRank),
          };
          const nodeWidth = metricWidths[sizeMetricRef.current];
          const nodeMass = isReferenced ? 2.8 : 1.05 + 1.5 * (degree / maxDegree);

          let finalColor: {
            background: string;
            border: string;
            highlight: { background: string; border: string };
            hover: { background: string; border: string };
          };
          let borderWidth: number, opacity: number;
          let shadow: boolean | Record<string, unknown> = false;
          if (!isHighlightMode) {
            const heatedBackground = pageRankHeatEnabled
              ? blendColors(gTheme.dimmedNodeBg, nodeColor, 0.24 + pageRankNorm * 0.76)
              : nodeColor;
            const heatedBorder = pageRankHeatEnabled
              ? blendColors('rgba(255,255,255,0.18)', gTheme.highlight, 0.12 + pageRankNorm * 0.6)
              : 'rgba(255,255,255,0.25)';
            finalColor = {
              background: heatedBackground,
              border: heatedBorder,
              highlight: { background: heatedBackground, border: gTheme.highlight },
              hover: {
                background: heatedBackground,
                border: pageRankHeatEnabled ? blendColors(heatedBorder, gTheme.highlight, 0.4) : 'rgba(255,255,255,0.6)',
              },
            };
            borderWidth = pageRankHeatEnabled ? 2 + pageRankNorm * 2.2 : 2;
            opacity = 1;
            if (pageRankHeatEnabled && pageRankNorm > 0.35) {
              shadow = {
                enabled: true,
                color: withAlpha(gTheme.highlight, 0.08 + pageRankNorm * 0.16),
                size: 10 + pageRankNorm * 18,
                x: 0,
                y: 0,
              };
            }
          } else if (isReferenced) {
            finalColor = {
              background: nodeColor,
              border: gTheme.highlight,
              highlight: { background: nodeColor, border: gTheme.highlight },
              hover: { background: nodeColor, border: gTheme.highlight },
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
            title: `${nodeType}: ${displayLabel}\n\nLabels: ${node.labels?.join(', ') || 'Unknown'}\nID: ${originalId}\nConnections: ${degree} (in ${inDegree} / out ${outDegree})\nPageRank: ${pageRank.toFixed(4)}\n\n${
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
              size: labelsRef.current ? (isReferenced ? 13 : 11) : 0,
              face: 'Helvetica, Arial, sans-serif',
              color: isHighlightMode && !isReferenced ? gTheme.nodeTextDimmed : gTheme.nodeText,
              strokeWidth: 0,
              align: 'center',
              vadjust: 0,
            },
            shape: 'circle',
            shadow,
            mass: nodeMass,
            _uniformWidth: metricWidths.uniform,
            _degreeWidth: metricWidths.degree,
            _inDegreeWidth: metricWidths.inDegree,
            _outDegreeWidth: metricWidths.outDegree,
            _pageRankWidth: metricWidths.pageRank,
            _baseColor: nodeColor,
          };
        });

      // Count highlighted nodes that survived filtering
      const visibleHighlightedCount = processedNodes.filter((n: { originalId: string | number }) =>
        referencedOriginalIds.has(n.originalId),
      ).length;
      dispatch({ type: 'SET_HIGHLIGHTED_COUNT', count: visibleHighlightedCount });

      const processedEdges = filteredRelationships
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
          const sourceReferenced = referencedOriginalIds.has(fromOrigId);
          const targetReferenced = referencedOriginalIds.has(toOrigId);
          const isHighlightedEdge = sourceReferenced && targetReferenced;
          const isDimmed = isHighlightMode && !isHighlightedEdge;
          const confWidth = confidence != null ? 0.8 + confidence * 2.2 : 1;
          const spacingScale = 0.72 + spacingRef.current * 1.28;
          const confSpringLength =
            confidence != null
              ? Math.round((210 - Math.min(Math.max(confidence, 0), 1) * 90) * spacingScale)
              : Math.round(180 * spacingScale);
          const confLabel = confidence != null ? `\nConfidence: ${(confidence * 100).toFixed(0)}%` : '';

          return {
            id: idCounterRef.current++,
            from: fromId,
            to: toId,
            label: relType,
            originalId: rel.id,
            arrows: { to: { enabled: true, scaleFactor: 0.8, type: 'arrow' } },
            color: {
              color: isDimmed ? gTheme.dimmedEdge : isHighlightedEdge ? gTheme.highlight : edgeColor,
              highlight: gTheme.highlight,
              hover: gTheme.highlight,
              opacity: isDimmed ? 0.25 : 1.0,
            },
            font: {
              size: labelsRef.current ? 11 : 0,
              face: 'Helvetica, Arial, sans-serif',
              color: isDimmed ? gTheme.nodeTextDimmed : isHighlightedEdge ? gTheme.highlight : gTheme.edgeText,
              background: gTheme.edgeLabelBg,
              strokeWidth: 0,
              align: 'top',
              vadjust: 15,
            },
            title: `${relType}\nFrom: ${fromOrigId}\nTo: ${toOrigId}${confLabel}`,
            width: isHighlightedEdge ? 3 : confWidth,
            length: isHighlightedEdge ? Math.max(120, confSpringLength - 20) : confSpringLength,
            _baseColor: edgeColor,
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
      const physicsSettings = getPhysicsSettings(isLargeGraph, spacingRef.current);

      const labelSize = labelsRef.current && !isLargeGraph;
      const options = {
        nodes: {
          shape: 'circle',
          size: isLargeGraph ? 20 : 30,
          font: {
            size: labelSize ? 12 : 0,
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
            size: labelSize ? 11 : 0,
            face: 'Helvetica, Arial, sans-serif',
            color: gTheme.edgeText,
            background: gTheme.edgeLabelBg,
            strokeWidth: 0,
            align: 'top',
            vadjust: 15,
          },
          color: { inherit: false },
          smooth: { enabled: !isLargeGraph, type: 'straightCross', forceDirection: false },
          shadow: false,
        },
        physics: physicsSettings,
        interaction: {
          hover: !isLargeGraph,
          tooltipDelay: 200,
          zoomView: true,
          dragView: true,
          dragNodes: true,
          multiselect: false,
          navigationButtons: false,
          keyboard: { enabled: true, bindToWindow: false },
          selectConnectedEdges: false,
        },
        layout: { improvedLayout: !isLargeGraph, hierarchical: false },
        configure: { enabled: false },
      };

      try {
        networkRef.current = new vis.Network(containerRef.current, { nodes, edges }, options);

        networkRef.current.on('stabilizationIterationsDone', () => {
          networkRef.current?.setOptions({
            physics: { enabled: physicsEnabledRef.current, stabilization: false },
          });
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
    [
      containerRef,
      networkRef,
      idCounterRef,
      initialViewRef,
      dispatch,
      onNodeClick,
      highlightedNodes,
      currentFilters,
      physicsEnabled,
      layoutSpacing,
    ],
  );

  useEffect(() => {
    if (graphData) {
      initializeGraph(graphData);
      if (searchTerm.trim() && applySearch) applySearch(searchTerm);
    } else if (networkRef.current) {
      networkRef.current.destroy();
      networkRef.current = null;
    }
  }, [graphData, initializeGraph]); // eslint-disable-line react-hooks/exhaustive-deps -- searchTerm/applySearch are intentionally excluded to avoid search-driven rebuilds

  return { initializeGraph };
}
