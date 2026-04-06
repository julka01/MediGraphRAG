import { useEffect, useCallback } from 'react';
import { getGraphTheme, normName, confidenceEdgeColor } from '../utils/graph-helpers';
import { generateNodeTypeColors, generateRelationshipTypeColors } from '../utils/colors';

export function useGraph(containerRef, appState, dispatch, networkRef, idCounterRef, initialViewRef, onNodeClick) {
  const {
    graphData,
    highlightedNodes,
    nodeTypeColors,
    relationshipTypeColors,
    currentFilters,
    showEdgeLabels,
  } = appState;

  const initializeGraph = useCallback((data) => {
    if (!data || (!data.nodes && !data.relationships)) return;
    if (!containerRef.current) return;

    const gTheme = getGraphTheme();

    // Pre-compute degree map
    const nodeDegreeMap = new Map();
    (data.relationships || []).forEach((rel) => {
      const f = rel.from || rel.start || rel.source;
      const t = rel.to || rel.end || rel.target;
      nodeDegreeMap.set(f, (nodeDegreeMap.get(f) || 0) + 1);
      nodeDegreeMap.set(t, (nodeDegreeMap.get(t) || 0) + 1);
    });

    // Destroy existing network
    if (networkRef.current) {
      try { networkRef.current.destroy(); } catch (e) { console.warn('Error destroying network:', e); }
      networkRef.current = null;
    }

    // Reset ID counter for fresh graph
    idCounterRef.current = 0;

    // Analyze types
    const nodeTypes = new Set();
    const relationshipTypes = new Set();
    (data.nodes || []).forEach((node) => {
      (node.labels?.length ? node.labels : ['Unknown']).forEach((l) => nodeTypes.add(l));
    });
    (data.relationships || []).forEach((rel) => {
      relationshipTypes.add(rel.type || 'Unknown');
    });

    // Generate colors
    const ntColors = generateNodeTypeColors(Array.from(nodeTypes));
    const rtColors = generateRelationshipTypeColors(Array.from(relationshipTypes));
    dispatch({ type: 'SET_NODE_TYPE_COLORS', colors: ntColors });
    dispatch({ type: 'SET_RELATIONSHIP_TYPE_COLORS', colors: rtColors });

    // Highlight analysis
    const normHighlighted = new Set([...highlightedNodes].map(normName));
    const referencedOriginalIds = new Set();
    if (normHighlighted.size > 0) {
      (data.nodes || []).forEach((node) => {
        const p = node.properties || {};
        const candidates = [p.name, p.id, p.title].filter(Boolean);
        if (candidates.some((c) => normHighlighted.has(normName(c)))) {
          referencedOriginalIds.add(node.id);
        }
      });
    }
    const isHighlightMode = referencedOriginalIds.size > 0;

    // ID mapping
    const nodeIdMap = new Map();

    // Process nodes
    const processedNodes = (data.nodes || [])
      .filter((node) => {
        if (currentFilters.nodeTypes.size > 0) {
          const nodeType = node.labels?.[0] || 'Unknown';
          return currentFilters.nodeTypes.has(nodeType);
        } else if (currentFilters.relationshipTypes.size > 0) {
          return data.relationships.some(
            (rel) =>
              currentFilters.relationshipTypes.has(rel.type || 'Unknown') &&
              ((rel.from || rel.start || rel.source) === node.id ||
                (rel.to || rel.end || rel.target) === node.id)
          );
        }
        return true;
      })
      .map((node) => {
        const originalId = node.id;
        let newId;
        if (nodeIdMap.has(originalId)) {
          newId = nodeIdMap.get(originalId);
        } else {
          newId = idCounterRef.current++;
          nodeIdMap.set(originalId, newId);
        }

        const nodeType = node.labels?.[0] || 'Unknown';
        const nodeColor = ntColors[nodeType] || '#428bca';

        let displayLabel = node.properties?.name || node.labels?.[0] || String(originalId);
        if (displayLabel.length > 20) displayLabel = displayLabel.substring(0, 17) + '...';

        const isReferenced = referencedOriginalIds.has(originalId);
        const degree = nodeDegreeMap.get(originalId) || 0;
        const degreeBonus = Math.min(degree * 2.5, 18);

        let finalColor, finalSize, borderWidth, opacity;
        if (!isHighlightMode) {
          finalColor = {
            background: nodeColor, border: 'rgba(255,255,255,0.25)',
            highlight: { background: nodeColor, border: '#ffd700' },
            hover: { background: nodeColor, border: 'rgba(255,255,255,0.6)' },
          };
          finalSize = 22 + degreeBonus;
          borderWidth = 2;
          opacity = 1;
        } else if (isReferenced) {
          finalColor = {
            background: nodeColor, border: '#ffd700',
            highlight: { background: nodeColor, border: '#ffaa00' },
            hover: { background: nodeColor, border: '#ffd700' },
          };
          finalSize = 30 + degreeBonus;
          borderWidth = 4;
          opacity = 1;
        } else {
          finalColor = {
            background: gTheme.dimmedNodeBg, border: gTheme.dimmedNodeBdr,
            highlight: { background: gTheme.dimmedNodeBdr, border: gTheme.dimmedNodeBdr },
            hover: { background: gTheme.dimmedNodeBg, border: gTheme.dimmedNodeBdr },
          };
          finalSize = 18 + degreeBonus;
          borderWidth = 1;
          opacity = 0.35;
        }

        return {
          ...node, id: newId, label: displayLabel, originalId,
          title: `${nodeType}: ${displayLabel}\n\nLabels: ${node.labels?.join(', ') || 'Unknown'}\nID: ${originalId}\n\n${node.properties ? Object.entries(node.properties).map(([k, v]) => `${k}: ${v}`).join('\n') : 'No properties'}`,
          color: finalColor, opacity, size: finalSize, borderWidth,
          font: {
            size: isReferenced ? 13 : 11,
            face: 'Helvetica, Arial, sans-serif',
            color: isHighlightMode && !isReferenced ? gTheme.nodeTextDimmed : gTheme.nodeText,
            strokeWidth: 0, align: 'center', vadjust: 0,
          },
          shape: 'circle', shadow: { enabled: false },
          _baseSize: 22 + degreeBonus, _baseColor: nodeColor,
        };
      });

    // Process edges
    const processedEdges = (data.relationships || [])
      .filter((rel) => {
        if (currentFilters.relationshipTypes.size > 0) {
          return currentFilters.relationshipTypes.has(rel.type || 'Unknown');
        }
        return true;
      })
      .map((rel) => {
        const fromOrigId = rel.from || rel.start || rel.source;
        const toOrigId = rel.to || rel.end || rel.target;
        const fromId = nodeIdMap.get(fromOrigId);
        const toId = nodeIdMap.get(toOrigId);
        if (fromId === undefined || toId === undefined) return null;

        const relType = rel.type || 'Unknown';
        const edgeColor = rtColors[relType] || '#888888';
        const confidence = rel.properties?.confidence ?? null;
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
          from: fromId, to: toId,
          label: showEdgeLabels ? relType : '',
          originalId: rel.id,
          arrows: { to: { enabled: true, scaleFactor: 0.8, type: 'arrow' } },
          color: {
            color: isDimmed ? gTheme.dimmedEdge : (isHighlightedEdge ? '#ffd700' : normalColor),
            highlight: '#ffd700', hover: '#ffd700',
            opacity: isDimmed ? 0.25 : 1.0,
          },
          font: {
            size: 11, face: 'Helvetica, Arial, sans-serif',
            color: isDimmed ? gTheme.nodeTextDimmed : (isHighlightedEdge ? '#ffd700' : gTheme.edgeText),
            background: gTheme.edgeLabelBg, strokeWidth: 0, align: 'top', vadjust: 15,
          },
          title: `${relType}\nFrom: ${fromOrigId}\nTo: ${toOrigId}${confLabel}`,
          width: isHighlightedEdge ? 3 : confWidth,
          _baseColor: normalColor, _baseWidth: confWidth,
          smooth: { enabled: true, type: 'straightCross', forceDirection: false },
          shadow: false,
        };
      })
      .filter(Boolean);

    if (processedNodes.length === 0) return;

    // Create network
    const nodes = new vis.DataSet(processedNodes);
    const edges = new vis.DataSet(processedEdges);

    const graphSize = processedNodes.length;
    const isLargeGraph = graphSize > 500;
    const isVeryLargeGraph = graphSize > 1000;

    let physicsSettings;
    if (isVeryLargeGraph) {
      physicsSettings = { enabled: false, stabilization: false };
    } else if (isLargeGraph) {
      physicsSettings = {
        enabled: true,
        stabilization: { enabled: true, iterations: 500, updateInterval: 25 },
        barnesHut: { gravitationalConstant: -2000, centralGravity: 0.3, springLength: 100, springConstant: 0.04, damping: 0.25, avoidOverlap: 0.02 },
        solver: 'barnesHut', minVelocity: 0.5, maxVelocity: 50, timestep: 0.4,
      };
    } else {
      physicsSettings = {
        enabled: true,
        stabilization: { enabled: true, iterations: 500, updateInterval: 25 },
        barnesHut: { gravitationalConstant: -5000, centralGravity: 0.3, springLength: 150, springConstant: 0.04, damping: 0.09, avoidOverlap: 0.1 },
        solver: 'barnesHut', minVelocity: 0.75, maxVelocity: 100, timestep: 0.35,
      };
    }

    const options = {
      nodes: {
        shape: 'circle', size: isLargeGraph ? 20 : 30,
        font: { size: isLargeGraph ? 0 : 12, face: 'Helvetica, Arial, sans-serif', color: gTheme.nodeText, strokeWidth: 0, align: 'center', vadjust: 0 },
        borderWidth: 1, borderWidthSelected: 3, shadow: false,
      },
      edges: {
        width: 1,
        arrows: { to: { enabled: true, scaleFactor: 0.8, type: 'arrow' } },
        font: { size: isLargeGraph ? 0 : 11, face: 'Helvetica, Arial, sans-serif', color: gTheme.edgeText, background: gTheme.edgeLabelBg, strokeWidth: 0, align: 'top', vadjust: 15 },
        color: { inherit: 'from' },
        smooth: { enabled: !isLargeGraph, type: 'straightCross', forceDirection: false },
        shadow: false,
      },
      physics: physicsSettings,
      interaction: {
        hover: !isLargeGraph, tooltipDelay: 200, zoomView: true, dragView: true,
        dragNodes: !isVeryLargeGraph, multiselect: false, navigationButtons: false,
        keyboard: true, selectConnectedEdges: false,
      },
      layout: { improvedLayout: !isLargeGraph, hierarchical: false },
      configure: { enabled: false },
    };

    try {
      networkRef.current = new vis.Network(containerRef.current, { nodes, edges }, options);

      networkRef.current.on('stabilizationIterationsDone', () => {
        networkRef.current.setOptions({ physics: { enabled: true, stabilization: false } });
        setTimeout(() => {
          try {
            initialViewRef.current = {
              scale: networkRef.current.getScale(),
              position: networkRef.current.getViewPosition(),
            };
          } catch { /* ignore */ }
        }, 1000);
      });

      networkRef.current.on('click', (params) => {
        if (params.nodes.length === 0) {
          onNodeClick?.(null);
          return;
        }
        const nodeId = params.nodes[0];
        const node = networkRef.current.body.data.nodes.get(nodeId);
        if (node) onNodeClick?.(node);
      });
    } catch (error) {
      console.error('Error creating network:', error);
    }
  }, [containerRef, networkRef, idCounterRef, initialViewRef, dispatch, onNodeClick, highlightedNodes, currentFilters, showEdgeLabels, nodeTypeColors, relationshipTypeColors]);

  // Re-initialize when key state changes
  useEffect(() => {
    if (graphData) {
      initializeGraph(graphData);
    }
  }, [graphData, currentFilters, highlightedNodes, showEdgeLabels]);

  return { initializeGraph };
}
