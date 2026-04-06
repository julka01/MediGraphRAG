import { state } from '../state.js';
import { getGraphTheme } from '../ui/theme.js';
import { showError } from '../ui/notifications.js';
import {
  generateNodeTypeColors,
  generateRelationshipTypeColors,
  updateOverview,
  updateMiniLegend,
  confidenceEdgeColor,
} from './legend.js';

export function initializeGraph(data, mergeMode = false) {

    if (!data || (!data.nodes && !data.relationships)) {
        console.log('No graph data provided');
        return;
    }

    const container = document.getElementById('graph-container');
    if (!container) {
        console.error('Graph container not found');
        return;
    }

    // Hide empty state when graph is rendered
    const emptyState = document.getElementById('graph-empty-state');
    if (emptyState) emptyState.style.display = 'none';

    const gTheme = getGraphTheme();

    // Pre-compute node degree (connection count) for size scaling
    const nodeDegreeMap = new Map();
    (data.relationships || []).forEach(rel => {
        const f = rel.from || rel.start || rel.source;
        const t = rel.to || rel.end || rel.target;
        nodeDegreeMap.set(f, (nodeDegreeMap.get(f) || 0) + 1);
        nodeDegreeMap.set(t, (nodeDegreeMap.get(t) || 0) + 1);
    });

    // If not in merge mode, completely clear any existing network and data
    if (!mergeMode) {
        if (state.network) {
            try {
                state.network.destroy();
                state.network = null;
            } catch (e) {
                console.warn('Error destroying network:', e);
                state.network = null;
            }
        }

        // Clear container but preserve overlay children (node-detail-panel, graph-empty-state)
        const preservedChildren = Array.from(container.children).filter(
            el => el.id === 'node-detail-panel' || el.id === 'graph-empty-state'
        );
        container.innerHTML = '';
        container.style.width = '100%';
        container.style.height = '100%';
        preservedChildren.forEach(el => container.appendChild(el));
    }

    // Analyze node and relationship types
    const nodeTypes = new Set();
    const relationshipTypes = new Set();

    (data.nodes || []).forEach(node => {
        if (node.labels && node.labels.length > 0) {
            node.labels.forEach(label => nodeTypes.add(label));
        } else {
            nodeTypes.add('Unknown');
        }
    });

    (data.relationships || []).forEach(rel => {
        if (rel.type) {
            relationshipTypes.add(rel.type);
        } else {
            relationshipTypes.add('Unknown');
        }
    });

    // Generate colors for types (only if not in merge mode)
    if (!mergeMode) {
        state.nodeTypeColors = generateNodeTypeColors(Array.from(nodeTypes));
        state.relationshipTypeColors = generateRelationshipTypeColors(Array.from(relationshipTypes));
        // Dynamic import to avoid circular dependency (filters.js imports from network.js)
        import('./filters.js').then(({ updateFilterPanel }) => {
            updateFilterPanel(state.nodeTypeColors, state.relationshipTypeColors);
        });
    }

    // Create a mapping from original Neo4j IDs to unique sequential IDs to avoid conflicts
    const nodeIdMap = new Map();

    // Normalise a string for name matching: lowercase + collapse underscores/spaces
    function normName(s) { return (s || '').toLowerCase().replace(/[_\s]+/g, ' ').trim(); }
    const normHighlighted = new Set([...state.highlightedNodes].map(normName));

    // Pre-compute which original node IDs are referenced so we can decide
    // whether to enter highlight mode only when at least one node matched.
    const referencedOriginalIds = new Set();
    if (normHighlighted.size > 0) {
        (data.nodes || []).forEach(node => {
            const p = node.properties || {};
            const candidates = [p.name, p.id, p.title].filter(Boolean);
            if (candidates.some(c => normHighlighted.has(normName(c)))) {
                referencedOriginalIds.add(node.id);
            }
        });
    }
    // Only dim when we actually matched something — otherwise show everything normally
    const isHighlightMode = referencedOriginalIds.size > 0;

    // Process nodes — show all, dim non-referenced ones when RAG has results
    const processedNodes = (data.nodes || [])
        .filter(node => {
            // Apply UI type filters (not RAG-related)
            if (state.currentFilters.nodeTypes.size > 0) {
                const nodeType = node.labels && node.labels.length > 0 ? node.labels[0] : 'Unknown';
                return state.currentFilters.nodeTypes.has(nodeType);
            } else if (state.currentFilters.relationshipTypes.size > 0) {
                return data.relationships.some(rel =>
                    state.currentFilters.relationshipTypes.has(rel.type || 'Unknown') &&
                    ((rel.from || rel.start || rel.source) === node.id ||
                     (rel.to || rel.end || rel.target) === node.id)
                );
            }
            return true;
        })
        .map(node => {
            const originalId = node.id;
            let newId;
            if (nodeIdMap.has(originalId)) {
                newId = nodeIdMap.get(originalId);
            } else {
                newId = state.uniqueIdCounter++;
                nodeIdMap.set(originalId, newId);
            }

            const nodeType = node.labels && node.labels.length > 0 ? node.labels[0] : 'Unknown';
            const nodeColor = state.nodeTypeColors[nodeType] || '#428bca';

            let displayLabel = (node.properties && node.properties.name) ||
                             (node.labels && node.labels[0]) ||
                             String(originalId);
            if (displayLabel.length > 20) {
                displayLabel = displayLabel.substring(0, 17) + '...';
            }

            const isReferenced = referencedOriginalIds.has(originalId);

            // Degree-based size scaling (always-on)
            const degree = nodeDegreeMap.get(originalId) || 0;
            const degreeBonus = Math.min(degree * 2.5, 18);

            let finalColor, finalSize, borderWidth, opacity;
            if (!isHighlightMode) {
                // No RAG answer yet (or no matches) — normal appearance
                finalColor = { background: nodeColor, border: 'rgba(255,255,255,0.25)',
                    highlight: { background: nodeColor, border: '#ffd700' },
                    hover: { background: nodeColor, border: 'rgba(255,255,255,0.6)' } };
                finalSize = 22 + degreeBonus;
                borderWidth = 2;
                opacity = 1;
            } else if (isReferenced) {
                // Referenced node — keep semantic type color, add gold ring
                finalColor = { background: nodeColor, border: '#ffd700',
                    highlight: { background: nodeColor, border: '#ffaa00' },
                    hover: { background: nodeColor, border: '#ffd700' } };
                finalSize = 30 + degreeBonus;
                borderWidth = 4;
                opacity = 1;
            } else {
                // Non-referenced node — dimmed
                finalColor = { background: gTheme.dimmedNodeBg, border: gTheme.dimmedNodeBdr,
                    highlight: { background: gTheme.dimmedNodeBdr, border: gTheme.dimmedNodeBdr },
                    hover: { background: gTheme.dimmedNodeBg, border: gTheme.dimmedNodeBdr } };
                finalSize = 18 + degreeBonus;
                borderWidth = 1;
                opacity = 0.35;
            }

            return {
                ...node,
                id: newId,
                label: displayLabel,
                originalId: originalId,
                title: `${nodeType}: ${displayLabel}\n\nLabels: ${node.labels ? node.labels.join(', ') : 'Unknown'}\nID: ${originalId}\n\n${node.properties ? Object.entries(node.properties).map(([k,v]) => `${k}: ${v}`).join('\n') : 'No properties'}`,
                color: finalColor,
                opacity: opacity,
                font: {
                    size: isReferenced ? 13 : 11,
                    face: 'Helvetica, Arial, sans-serif',
                    color: isHighlightMode && !isReferenced ? gTheme.nodeTextDimmed : gTheme.nodeText,
                    strokeWidth: 0,
                    align: 'center',
                    vadjust: 0
                },
                size: finalSize,
                borderWidth: borderWidth,
                shape: 'circle',
                shadow: { enabled: false },
                _baseSize: 22 + degreeBonus,
                _baseColor: nodeColor
            };
        });

    // Process relationships — show all, dim when RAG highlights active
    const processedEdges = (data.relationships || [])
        .filter(rel => {
            const relType = rel.type || 'Unknown';
            if (state.currentFilters.relationshipTypes.size > 0) {
                return state.currentFilters.relationshipTypes.has(relType);
            }

            // No filters active - show all relationships (but associated nodes might still be filtered above)
            return true;
        })
        .map(rel => {
            const fromOrigId = rel.from || rel.start || rel.source;
            const toOrigId = rel.to || rel.end || rel.target;
            const fromId = nodeIdMap.get(fromOrigId);
            const toId = nodeIdMap.get(toOrigId);

            if (fromId === undefined || toId === undefined) return null;

            const relType = rel.type || 'Unknown';
            const edgeColor = state.relationshipTypeColors[relType] || '#888888';

            // Confidence-based colour (stored on rel.properties.confidence by triple verification)
            const confidence = rel.properties && rel.properties.confidence != null
                ? rel.properties.confidence : null;
            const confColor = confidenceEdgeColor(confidence);

            // Highlight edge if both endpoints are referenced nodes
            const sourceReferenced = referencedOriginalIds.has(fromOrigId);
            const targetReferenced = referencedOriginalIds.has(toOrigId);
            const isHighlightedEdge = sourceReferenced && targetReferenced;
            const isDimmed = isHighlightMode && !isHighlightedEdge;

            // Normal colour: use confidence-coded colour when available, else type colour
            const normalColor = confColor || edgeColor;
            // Width: high confidence gets a slightly thicker line (1.5→2.5 range)
            const confWidth = confidence != null ? (0.8 + confidence * 2.2) : 1;

            const confLabel = confidence != null
                ? `\nConfidence: ${(confidence * 100).toFixed(0)}%` : '';

            return {
                id: state.uniqueIdCounter++,
                from: fromId,
                to: toId,
                label: state.showEdgeLabels ? relType : '',
                originalId: rel.id,
                arrows: { to: { enabled: true, scaleFactor: 0.8, type: 'arrow' } },
                color: {
                    color: isDimmed ? gTheme.dimmedEdge : (isHighlightedEdge ? '#ffd700' : normalColor),
                    highlight: '#ffd700',
                    hover: '#ffd700',
                    opacity: isDimmed ? 0.25 : 1.0
                },
                font: {
                    size: 11,
                    face: 'Helvetica, Arial, sans-serif',
                    color: isDimmed ? gTheme.nodeTextDimmed : (isHighlightedEdge ? '#ffd700' : gTheme.edgeText),
                    background: gTheme.edgeLabelBg,
                    strokeWidth: 0,
                    align: 'top',
                    vadjust: 15
                },
                title: `${relType}\nFrom: ${fromOrigId}\nTo: ${toOrigId}${confLabel}`,
                width: isHighlightedEdge ? 3 : confWidth,
                _baseColor: normalColor,
                _baseWidth: confWidth,
                smooth: { enabled: true, type: 'straightCross', forceDirection: false },
                shadow: false
            };
        }).filter(edge => edge !== null);

    console.log(`Processed ${processedNodes.length} nodes and ${processedEdges.length} edges`);

    if (processedNodes.length === 0) {
        console.log('No nodes to display after filtering');
        return;
    }

    if (mergeMode && state.network) {
        // Merge mode: Add new nodes and edges to existing graph
        console.log('Merging with existing graph...');

        // Get existing node IDs to avoid conflicts
        const existingNodeIds = new Set();
        state.network.body.data.nodes.get().forEach(node => existingNodeIds.add(node.id));

        // Filter out nodes that already exist
        const newNodesOnly = processedNodes.filter(node => !existingNodeIds.has(node.id));
        console.log(`Filtered ${processedNodes.length - newNodesOnly.length} duplicate nodes, adding ${newNodesOnly.length} new nodes`);

        // Add new nodes only
        if (newNodesOnly.length > 0) {
            try {
                state.network.body.data.nodes.add(newNodesOnly);
                console.log(`Added ${newNodesOnly.length} new nodes`);
            } catch (e) {
                console.error('Error adding nodes in merge mode:', e);
            }
        }

        // Get existing edge IDs to avoid conflicts
        const existingEdgeIds = new Set();
        state.network.body.data.edges.get().forEach(edge => existingEdgeIds.add(edge.id));

        // Filter out edges that already exist
        const newEdgesOnly = processedEdges.filter(edge => !existingEdgeIds.has(edge.id));
        console.log(`Filtered ${processedEdges.length - newEdgesOnly.length} duplicate edges, adding ${newEdgesOnly.length} new edges`);

        // Add new edges only
        if (newEdgesOnly.length > 0) {
            try {
                state.network.body.data.edges.add(newEdgesOnly);
                console.log(`Added ${newEdgesOnly.length} new edges`);
            } catch (e) {
                console.error('Error adding edges in merge mode:', e);
            }
        }

        // Update stored graph data
        if (!state.graphData) state.graphData = { nodes: [], relationships: [] };
        state.graphData.nodes = [...(state.graphData.nodes || []), ...newNodesOnly];
        state.graphData.relationships = [...(state.graphData.relationships || []), ...newEdgesOnly];

    } else {
        // Replace mode: Create new graph
        console.log('Creating new graph...');

        const nodes = new vis.DataSet(processedNodes);
        const edges = new vis.DataSet(processedEdges);

        // Performance optimizations based on graph size
        const graphSize = processedNodes.length;
        const isLargeGraph = graphSize > 500;
        const isVeryLargeGraph = graphSize > 1000;

        // Adaptive physics settings for performance
        let physicsSettings;
        if (isVeryLargeGraph) {
            // Minimal physics for very large graphs
            physicsSettings = {
                enabled: false, // Disable physics for very large graphs
                stabilization: false
            };
        } else if (isLargeGraph) {
            // Optimized physics for large graphs - improved stability
            physicsSettings = {
                enabled: true,
                stabilization: {
                    enabled: true,
                    iterations: 500,
                    updateInterval: 25
                },
                barnesHut: {
                    gravitationalConstant: -2000,
                    centralGravity: 0.3,
                    springLength: 100,
                    springConstant: 0.04,
                    damping: 0.25,
                    avoidOverlap: 0.02
                },
                solver: 'barnesHut',
                minVelocity: 0.5,
                maxVelocity: 50,
                timestep: 0.4
            };
        } else {
            // Full physics for small/medium graphs
            physicsSettings = {
                enabled: true,
                stabilization: {
                    enabled: true,
                    iterations: 500,
                    updateInterval: 25
                },
                barnesHut: {
                    gravitationalConstant: -5000,
                    centralGravity: 0.3,
                    springLength: 150,
                    springConstant: 0.04,
                    damping: 0.09,
                    avoidOverlap: 0.1
                },
                solver: 'barnesHut',
                minVelocity: 0.75,
                maxVelocity: 100,
                timestep: 0.35
            };
        }

        // Update overview panel with current graph data (only if not in merge mode)
        if (!mergeMode) {
            updateOverview(data);
        }

        const options = {
            nodes: {
                shape: 'circle', // Neo4j style circles
                size: isLargeGraph ? 20 : 30, // Smaller nodes for large graphs
                font: {
                    size: isLargeGraph ? 0 : 12,
                    face: 'Helvetica, Arial, sans-serif',
                    color: gTheme.nodeText,
                    strokeWidth: 0,
                    align: 'center',
                    vadjust: 0
                },
                borderWidth: 1,
                borderWidthSelected: 3,
                shadow: false
            },
            edges: {
                width: 1,
                arrows: {
                    to: {
                        enabled: true,
                        scaleFactor: 0.8,
                        type: 'arrow'
                    }
                },
                font: {
                    size: isLargeGraph ? 0 : 11,
                    face: 'Helvetica, Arial, sans-serif',
                    color: gTheme.edgeText,
                    background: gTheme.edgeLabelBg,
                    strokeWidth: 0,
                    align: 'top',
                    vadjust: 15
                },
                color: {
                    inherit: 'from'
                },
                smooth: {
                    enabled: !isLargeGraph, // Disable smooth curves on large graphs
                    type: 'straightCross',
                    forceDirection: false
                },
                shadow: false
            },
            physics: physicsSettings,
            interaction: {
                hover: !isLargeGraph, // Disable hover on large graphs for performance
                tooltipDelay: 200,
                zoomView: true,
                dragView: true,
                dragNodes: !isVeryLargeGraph, // Disable node dragging on very large graphs
                multiselect: false,
                navigationButtons: false,
                keyboard: true,
                selectConnectedEdges: false
            },
            layout: {
                improvedLayout: !isLargeGraph,
                hierarchical: false
            },
            configure: {
                enabled: false
            }
        };

        const containerElement = document.getElementById('graph-container');
        console.log('Creating network with container:', containerElement);
        console.log('Nodes data:', nodes.length, 'Edges data:', edges.length);

        try {
            state.network = new vis.Network(containerElement, { nodes, edges }, options);
            console.log('Network created successfully');
            updateMiniLegend(state.nodeTypeColors);

            // Add stabilization callback
            state.network.on('stabilizationIterationsDone', function() {
                console.log('Graph stabilization complete');

                // Re-enable physics for dynamic behavior after stabilization
                state.network.setOptions({
                    physics: {
                        enabled: true,
                        stabilization: false
                    }
                });

                // Save initial view state after physics settles and first render completes
                setTimeout(() => {
                    try {
                        const currentPos = state.network.getViewPosition();
                        const currentScale = state.network.getScale();

                        console.log('Current position object:', JSON.stringify(currentPos));
                        console.log('Current scale:', currentScale);
                        console.log('Position x:', currentPos ? currentPos.x : 'undefined');
                        console.log('Position y:', currentPos ? currentPos.y : 'undefined');

                        state.initialViewState = {
                            scale: currentScale,
                            position: currentPos
                        };
                        console.log('Initial view state saved:', JSON.stringify(state.initialViewState));
                    } catch (error) {
                        console.error('Error saving initial view state:', error);
                    }
                }, 1000);
            });

        } catch (error) {
            console.error('Error creating network:', error);
            showError(`Failed to create graph visualization: ${error.message}`);
            return;
        }
    }

    // Node click → show detail side panel
    // `nodes` is the local vis.DataSet defined above in the replace-mode branch.
    // When in merge mode, state.network already has a click handler from the original creation.
    state.network.on("click", function (params) {
        const panel = document.getElementById('node-detail-panel');
        if (!panel) return;

        if (params.nodes.length === 0) {
            // Click on empty canvas — close panel
            panel.classList.remove('open');
            return;
        }

        const nodeId = params.nodes[0];
        const node = state.network.body.data.nodes.get(nodeId);
        if (!node) return;

        const nodeType = node.labels && node.labels.length > 0 ? node.labels[0] : 'Unknown';
        const nodeColor = state.nodeTypeColors[nodeType] || '#428bca';

        document.getElementById('node-panel-name').textContent = node.label || node.originalId;

        const typeEl = document.getElementById('node-panel-type');
        typeEl.textContent = nodeType;
        typeEl.style.background = nodeColor + '33';
        typeEl.style.color = nodeColor;
        typeEl.style.border = '1px solid ' + nodeColor + '55';

        const propsEl = document.getElementById('node-panel-props');
        propsEl.innerHTML = '';
        if (node.properties && Object.keys(node.properties).length > 0) {
            Object.entries(node.properties).forEach(([k, v]) => {
                const row = document.createElement('div');
                row.className = 'nd-prop';
                const key = document.createElement('span');
                key.className = 'nd-key';
                key.textContent = k;
                const val = document.createElement('span');
                val.className = 'nd-val';
                val.textContent = v;
                row.appendChild(key);
                row.appendChild(val);
                propsEl.appendChild(row);
            });
        } else {
            const empty = document.createElement('div');
            empty.style.cssText = 'color:var(--text-3);font-size:11px;';
            empty.textContent = 'No additional properties';
            propsEl.appendChild(empty);
        }

        panel.classList.add('open');
    });
}
