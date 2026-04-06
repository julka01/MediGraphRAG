import { state } from '../state.js';
import { initializeGraph } from './network.js';

// Update filter panel with checkboxes for node types and relationship types
export function updateFilterPanel(nodeTypes, relationshipTypes) {
    const nodeTypeFilters = document.getElementById('node-type-filters');
    const relationshipTypeFilters = document.getElementById('relationship-type-filters');

    nodeTypeFilters.innerHTML = '';
    relationshipTypeFilters.innerHTML = '';

    // Count nodes per type from state.graphData
    const nodeCounts = {};
    const relCounts = {};
    if (state.graphData) {
        (state.graphData.nodes || []).forEach(n => {
            const t = n.type || n.label || 'Unknown';
            nodeCounts[t] = (nodeCounts[t] || 0) + 1;
        });
        (state.graphData.relationships || []).forEach(r => {
            const t = r.type || 'Unknown';
            relCounts[t] = (relCounts[t] || 0) + 1;
        });
    }

    // Add node type filters
    Object.keys(nodeTypes).forEach(type => {
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `node-${type}`;
        checkbox.className = 'filter-checkbox';
        checkbox.checked = true;
        checkbox.addEventListener('change', () => applyFilters());

        const label = document.createElement('label');
        label.htmlFor = `node-${type}`;
        label.textContent = type;

        const count = nodeCounts[type];
        if (count !== undefined) {
            const badge = document.createElement('span');
            badge.className = 'filter-count';
            badge.textContent = count;
            label.appendChild(badge);
        }

        const div = document.createElement('div');
        div.appendChild(checkbox);
        div.appendChild(label);
        nodeTypeFilters.appendChild(div);
    });

    // Add relationship type filters
    Object.keys(relationshipTypes).forEach(type => {
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `rel-${type}`;
        checkbox.className = 'filter-checkbox';
        checkbox.checked = true;
        checkbox.addEventListener('change', () => applyFilters());

        const label = document.createElement('label');
        label.htmlFor = `rel-${type}`;
        label.textContent = type;

        const count = relCounts[type];
        if (count !== undefined) {
            const badge = document.createElement('span');
            badge.className = 'filter-count';
            badge.textContent = count;
            label.appendChild(badge);
        }

        const div = document.createElement('div');
        div.appendChild(checkbox);
        div.appendChild(label);
        relationshipTypeFilters.appendChild(div);
    });
}

// Read checkbox states, update state.currentFilters, and re-render the graph
export function applyFilters() {
    if (!state.network || !state.graphData) return;

    const nodeTypeCheckboxes = document.querySelectorAll('#node-type-filters input[type="checkbox"]');
    const relationshipTypeCheckboxes = document.querySelectorAll('#relationship-type-filters input[type="checkbox"]');

    // Update current filters
    state.currentFilters.nodeTypes.clear();
    state.currentFilters.relationshipTypes.clear();

    nodeTypeCheckboxes.forEach(checkbox => {
        if (checkbox.checked) {
            const type = checkbox.id.replace('node-', '');
            state.currentFilters.nodeTypes.add(type);
        }
    });

    relationshipTypeCheckboxes.forEach(checkbox => {
        if (checkbox.checked) {
            const type = checkbox.id.replace('rel-', '');
            state.currentFilters.relationshipTypes.add(type);
        }
    });

    // Re-initialize graph with filters
    initializeGraph(state.graphData);
}

// Dims non-matching nodes and edges; highlights matching ones
export function performSearch(searchTerm) {
    if (!state.network) return;

    const nodeDS = state.network.body.data.nodes;
    const edgeDS = state.network.body.data.edges;
    const term   = (searchTerm || '').toLowerCase().trim();

    if (!term) {
        // Clear — restore full opacity
        nodeDS.update(nodeDS.get().map(n => ({ id: n.id, opacity: 1, hidden: false })));
        edgeDS.update(edgeDS.get().map(e => ({ id: e.id, opacity: 1, hidden: false })));
        const clr = document.getElementById('search-clear');
        if (clr) clr.style.display = 'none';
        const countEl = document.getElementById('search-count');
        if (countEl) countEl.style.display = 'none';
        state.network.redraw();
        return;
    }

    const matched = new Set();
    const nodeUpdates = nodeDS.get().map(node => {
        const hit = (node.label || '').toLowerCase().includes(term)
                 || (node.title || '').toLowerCase().includes(term)
                 || JSON.stringify(node.properties || {}).toLowerCase().includes(term);
        if (hit) matched.add(node.id);
        // Highlight matches; dim non-matches instead of hiding
        return { id: node.id, hidden: false, opacity: hit ? 1 : 0.1 };
    });
    nodeDS.update(nodeUpdates);

    // Dim edges with no matched endpoint, keep edges between matched nodes
    const edgeUpdates = edgeDS.get().map(edge => ({
        id: edge.id,
        hidden: false,
        opacity: (matched.has(edge.from) || matched.has(edge.to)) ? 1 : 0.06
    }));
    edgeDS.update(edgeUpdates);

    const clr = document.getElementById('search-clear');
    if (clr) clr.style.display = 'block';

    const countEl = document.getElementById('search-count');
    if (countEl) {
        countEl.textContent = `${matched.size} match${matched.size === 1 ? '' : 'es'}`;
        countEl.style.display = matched.size > 0 ? 'block' : 'none';
    }
    state.network.redraw();
}

// Wire up all filter and search-related event listeners
export function initFilters() {
    document.getElementById('node-search').addEventListener('input', function(e) {
        performSearch(e.target.value);
    });

    document.getElementById('search-clear').addEventListener('click', function() {
        document.getElementById('node-search').value = '';
        performSearch('');
    });

    document.getElementById('filter-btn').addEventListener('click', function() {
        const filterPanel = document.getElementById('filter-panel');
        filterPanel.style.display = filterPanel.style.display === 'none' ? 'block' : 'none';
    });

    document.getElementById('apply-filters').addEventListener('click', function() {
        applyFilters();
        document.getElementById('filter-panel').style.display = 'none';
    });

    document.getElementById('reset-filters').addEventListener('click', function() {
        // Reset all checkboxes to checked
        const checkboxes = document.querySelectorAll('#filter-panel input[type="checkbox"]');
        checkboxes.forEach(checkbox => checkbox.checked = true);

        // Clear current filters and reset to "All"
        state.currentFilters.nodeTypes.clear();
        state.currentFilters.relationshipTypes.clear();

        // Re-initialize graph
        if (state.graphData) {
            initializeGraph(state.graphData);
        }

        // Reset checkboxes to checked state
        const nodeCheckboxes = document.querySelectorAll('#node-type-filters input[type="checkbox"]');
        const relCheckboxes = document.querySelectorAll('#relationship-type-filters input[type="checkbox"]');
        nodeCheckboxes.forEach(checkbox => checkbox.checked = true);
        relCheckboxes.forEach(checkbox => checkbox.checked = true);

        document.getElementById('filter-panel').style.display = 'none';
    });

    document.getElementById('edge-labels').addEventListener('change', function() {
        state.showEdgeLabels = this.checked;
        if (state.graphData) initializeGraph(state.graphData);
    });
}
