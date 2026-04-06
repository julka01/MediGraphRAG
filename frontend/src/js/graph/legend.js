import { state } from '../state.js';
import { getGraphTheme } from '../ui/theme.js';

// Generate a distinct color for each node type using a fixed 24-color palette.
export function generateNodeTypeColors(nodeTypes) {
    const colors = [
        '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
        '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928',
        '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
        '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f'
    ];

    const colorMap = {};
    nodeTypes.forEach((type, index) => {
        colorMap[type] = colors[index % colors.length];
    });
    return colorMap;
}

// Generate a grey-tone color for each relationship type using a fixed 10-color palette.
export function generateRelationshipTypeColors(relationshipTypes) {
    const colors = [
        '#444444', '#666666', '#888888', '#aaaaaa', '#cccccc',
        '#2c3e50', '#34495e', '#7f8c8d', '#95a5a6', '#bdc3c7'
    ];

    const colorMap = {};
    relationshipTypes.forEach((type, index) => {
        colorMap[type] = colors[index % colors.length];
    });
    return colorMap;
}

// Update the #legend-panel DOM element with node type and relationship type color swatches.
export function updateLegend(nodeTypes, relationshipTypes) {
    const legendPanel = document.getElementById('legend-panel');
    const legendContent = document.getElementById('legend-content');

    if (!legendPanel) return; // If legend panel doesn't exist, skip

    legendContent.innerHTML = '';

    // Add node type legends
    for (const [type, color] of Object.entries(nodeTypes)) {
        const legendItem = document.createElement('div');
        legendItem.className = 'legend-item';
        const swatch = document.createElement('span');
        swatch.className = 'legend-color-box';
        swatch.style.backgroundColor = color;
        const label = document.createElement('span');
        label.className = 'legend-text';
        label.textContent = type;
        legendItem.appendChild(swatch);
        legendItem.appendChild(label);
        legendContent.appendChild(legendItem);
    }

    // Add separator if we have both node and relationship types
    if (Object.keys(relationshipTypes).length > 0) {
        const separator = document.createElement('hr');
        separator.style.cssText = 'margin:10px 0;border:none;border-top:1px solid #ddd;';
        legendContent.appendChild(separator);
    }

    // Add relationship type legends
    for (const [type, color] of Object.entries(relationshipTypes)) {
        const legendItem = document.createElement('div');
        legendItem.className = 'legend-item';
        const swatch = document.createElement('span');
        swatch.className = 'legend-relation-box';
        swatch.style.cssText = `border-color:${color};background-color:rgba(255,255,255,0.8);`;
        const label = document.createElement('span');
        label.className = 'legend-text';
        label.textContent = type + ' →';
        legendItem.appendChild(swatch);
        legendItem.appendChild(label);
        legendContent.appendChild(legendItem);
    }

    // Show legend panel if we have content
    if (Object.keys(nodeTypes).length > 0 || Object.keys(relationshipTypes).length > 0) {
        legendPanel.style.display = 'block';
    } else {
        legendPanel.style.display = 'none';
    }
}

// Update the #overview-panel with node label counts and relationship type counts.
// Click handlers on overview items trigger a filtered graph re-render via dynamic import
// to avoid a circular dependency with graph/network.js.
export function updateOverview(data) {
    if (!data) {
        document.getElementById('overview-panel').style.display = 'none';
        return;
    }

    const nodeLabelsOverview = document.getElementById('node-labels-overview');
    const relationshipTypesOverview = document.getElementById('relationship-types-overview');

    // Count node labels
    const nodeLabelCounts = {};
    (data.nodes || []).forEach(node => {
        const label = node.labels && node.labels.length > 0 ? node.labels[0] : 'Unknown';
        nodeLabelCounts[label] = (nodeLabelCounts[label] || 0) + 1;
    });

    // Count relationship types
    const relationshipTypeCounts = {};
    (data.relationships || []).forEach(rel => {
        const type = rel.type || 'Unknown';
        relationshipTypeCounts[type] = (relationshipTypeCounts[type] || 0) + 1;
    });

    // Update node labels overview
    nodeLabelsOverview.innerHTML = '';
    Object.entries(nodeLabelCounts)
        .sort(([,a], [,b]) => b - a) // Sort by count descending
        .forEach(([label, count]) => {
            const item = document.createElement('div');
            item.className = 'overview-item';
            const color = state.nodeTypeColors[label] || '#428bca';
            const labelRow = document.createElement('div');
            labelRow.className = 'overview-item-label';
            const swatch = document.createElement('div');
            swatch.className = 'overview-color';
            swatch.style.backgroundColor = color;
            const nameSpan = document.createElement('span');
            nameSpan.textContent = label;
            labelRow.appendChild(swatch);
            labelRow.appendChild(nameSpan);
            const countDiv = document.createElement('div');
            countDiv.className = 'overview-count';
            countDiv.textContent = count;
            item.appendChild(labelRow);
            item.appendChild(countDiv);
            item.addEventListener('click', () => {
                // Exclusive filtering: clear all other filters and set only this node type
                state.currentFilters.nodeTypes.clear();
                state.currentFilters.relationshipTypes.clear();
                if (state.currentFilters.nodeTypes.has(label)) {
                    state.currentFilters.nodeTypes.delete(label);
                } else {
                    state.currentFilters.nodeTypes.add(label);
                }
                if (state.graphData) {
                    import('../graph/network.js').then(({ initializeGraph }) => initializeGraph(state.graphData));
                }
            });
            nodeLabelsOverview.appendChild(item);
        });

    // Update relationship types overview
    relationshipTypesOverview.innerHTML = '';
    Object.entries(relationshipTypeCounts)
        .sort(([,a], [,b]) => b - a) // Sort by count descending
        .forEach(([type, count]) => {
            const item = document.createElement('div');
            item.className = 'overview-item';
            const color = state.relationshipTypeColors[type] || '#888888';
            const labelRow = document.createElement('div');
            labelRow.className = 'overview-item-label';
            const swatch = document.createElement('div');
            swatch.className = 'overview-color';
            swatch.style.backgroundColor = color;
            const nameSpan = document.createElement('span');
            nameSpan.textContent = type;
            labelRow.appendChild(swatch);
            labelRow.appendChild(nameSpan);
            const countDiv = document.createElement('div');
            countDiv.className = 'overview-count';
            countDiv.textContent = count;
            item.appendChild(labelRow);
            item.appendChild(countDiv);
            item.addEventListener('click', () => {
                // Exclusive filtering: clear all other filters and set only this relationship type
                state.currentFilters.nodeTypes.clear();
                state.currentFilters.relationshipTypes.clear();
                if (state.currentFilters.relationshipTypes.has(type)) {
                    state.currentFilters.relationshipTypes.delete(type);
                } else {
                    state.currentFilters.relationshipTypes.add(type);
                }
                if (state.graphData) {
                    import('../graph/network.js').then(({ initializeGraph }) => initializeGraph(state.graphData));
                }
            });
            relationshipTypesOverview.appendChild(item);
        });

    // Show overview panel
    document.getElementById('overview-panel').style.display = 'block';

    // Set up overview header toggle
    const overviewHeader = document.getElementById('overview-header');
    const overviewToggle = document.getElementById('overview-toggle');
    const overviewContent = document.getElementById('overview-content');

    overviewHeader.addEventListener('click', function() {
        const isCollapsed = overviewContent.classList.contains('collapsed');
        if (isCollapsed) {
            overviewContent.classList.remove('collapsed');
            overviewContent.classList.add('expanded');
            overviewToggle.classList.remove('collapsed');
        } else {
            overviewContent.classList.remove('expanded');
            overviewContent.classList.add('collapsed');
            overviewToggle.classList.add('collapsed');
        }
    });

}

// Show or hide the #active-kg-badge with the given KG name.
export function updateKGBadge(name) {
    const badge = document.getElementById('active-kg-badge');
    if (!badge) return;
    if (name) {
        badge.textContent = name;
        badge.style.display = 'inline-block';
    } else {
        badge.style.display = 'none';
    }
}

// Update the #chat-kg-name element with the given KG name.
export function updateChatKGName(name) {
    const el = document.getElementById('chat-kg-name');
    if (!el) return;
    if (name) {
        el.textContent = '— ' + name;
        el.style.display = 'inline';
    } else {
        el.style.display = 'none';
    }
}

// Update the #highlight-badge and #hl-badge-text based on the highlighted entity count.
export function updateHighlightBadge(count) {
    const badge = document.getElementById('highlight-badge');
    const text  = document.getElementById('hl-badge-text');
    if (!badge || !text) return;
    if (count > 0) {
        text.textContent = `${count} ${count === 1 ? 'entity' : 'entities'} selected/highlighted  `;
        badge.style.display = 'block';
    } else {
        badge.style.display = 'none';
    }
}

// Update #mini-legend with up to 14 color dots for the given type→color map.
export function updateMiniLegend(typeColors) {
    const legend = document.getElementById('mini-legend');
    const items  = document.getElementById('mini-legend-items');
    if (!legend || !items) return;
    const entries = Object.entries(typeColors);
    if (entries.length === 0) { legend.style.display = 'none'; return; }
    items.innerHTML = entries.slice(0, 14).map(([type, color]) =>
        `<div class="mini-legend-item"><span class="mini-legend-dot" style="background:${color}"></span><span>${type}</span></div>`
    ).join('');
    legend.style.display = 'block';
}

// Map a confidence value (0.1–1.0) to an edge colour: gold → amber → grey → dim.
export function confidenceEdgeColor(conf) {
    if (conf == null) return null;
    if (conf >= 0.8) return '#f1c40f';   // gold   — high confidence
    if (conf >= 0.5) return '#e67e22';   // amber  — medium confidence
    if (conf >= 0.3) return '#95a5a6';   // grey   — low confidence
    return '#555e68';                     // dim    — very low confidence
}

// Update node/edge styles in-place when RAG highlights entities.
// Preserves the current graph layout and physics state without rebuilding the network.
export function applyHighlightStyles() {
    if (!state.network) return;
    const gTheme = getGraphTheme();
    function normName(s) { return (s || '').toLowerCase().replace(/[_\s]+/g, ' ').trim(); }
    const normHighlighted = new Set([...state.highlightedNodes].map(normName));

    // Find which network node IDs are referenced
    const referencedIds = new Set();
    if (normHighlighted.size > 0) {
        state.network.body.data.nodes.get().forEach(node => {
            const p = node.properties || {};
            const candidates = [p.name, p.id, p.title].filter(Boolean);
            if (candidates.some(c => normHighlighted.has(normName(c)))) {
                referencedIds.add(node.id);
            }
        });
    }
    const isHighlightMode = referencedIds.size > 0;

    const nodeUpdates = state.network.body.data.nodes.get().map(node => {
        const baseColor = node._baseColor || state.nodeTypeColors[(node.labels && node.labels[0]) || 'Unknown'] || '#428bca';
        const baseSize  = node._baseSize  || 22;
        const isRef     = referencedIds.has(node.id);
        let color, borderWidth, opacity, fontSize;
        if (!isHighlightMode) {
            color = { background: baseColor, border: 'rgba(255,255,255,0.25)',
                highlight: { background: baseColor, border: '#ffd700' },
                hover:     { background: baseColor, border: 'rgba(255,255,255,0.6)' } };
            borderWidth = 2; opacity = 1; fontSize = 11;
        } else if (isRef) {
            color = { background: baseColor, border: '#ffd700',
                highlight: { background: baseColor, border: '#ffaa00' },
                hover:     { background: baseColor, border: '#ffd700' } };
            borderWidth = 4; opacity = 1; fontSize = 13;
        } else {
            color = { background: gTheme.dimmedNodeBg, border: gTheme.dimmedNodeBdr,
                highlight: { background: gTheme.dimmedNodeBdr, border: gTheme.dimmedNodeBdr },
                hover:     { background: gTheme.dimmedNodeBg, border: gTheme.dimmedNodeBdr } };
            borderWidth = 1; opacity = 0.35; fontSize = 11;
        }
        return {
            id: node.id, color, size: isRef ? baseSize + 8 : baseSize,
            borderWidth, opacity,
            font: { ...node.font, size: fontSize,
                color: isHighlightMode && !isRef ? gTheme.nodeTextDimmed : gTheme.nodeText }
        };
    });

    const edgeUpdates = state.network.body.data.edges.get().map(edge => {
        const srcRef = referencedIds.has(edge.from);
        const tgtRef = referencedIds.has(edge.to);
        const isHlEdge = srcRef && tgtRef;
        const isDimmed = isHighlightMode && !isHlEdge;
        const origColor = edge._baseColor || '#888888';
        return {
            id: edge.id,
            color: { color: isDimmed ? gTheme.dimmedEdge : (isHlEdge ? '#ffd700' : origColor),
                highlight: '#ffd700', hover: '#ffd700', opacity: isDimmed ? 0.25 : 1.0 },
            width: isHlEdge ? 3 : (edge._baseWidth || 1),
            font: { ...edge.font,
                color: isDimmed ? gTheme.nodeTextDimmed : (isHlEdge ? '#ffd700' : gTheme.edgeText) }
        };
    });

    state.network.body.data.nodes.update(nodeUpdates);
    state.network.body.data.edges.update(edgeUpdates);
}
