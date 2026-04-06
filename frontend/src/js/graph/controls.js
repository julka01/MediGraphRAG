import { state } from '../state.js';
import { showError, showSuccess } from '../ui/notifications.js';
import { initializeGraph } from './network.js';
import { applyHighlightStyles, updateHighlightBadge } from './legend.js';

// Create advanced toggle button in .graph-controls if not already present
export function createAdvancedButton() {
    const controlsDiv = document.querySelector('.graph-controls');
    if (controlsDiv && !document.getElementById('advanced-toggle')) {
        const advancedBtn = document.createElement('button');
        advancedBtn.id = 'advanced-toggle';
        advancedBtn.className = 'zoom-btn';
        advancedBtn.textContent = '⚙️ Advanced';
        advancedBtn.title = 'Show Advanced Visualization Controls';
        advancedBtn.addEventListener('click', () => {
            const advancedControls = document.getElementById('advanced-controls');
            advancedControls.style.display = advancedControls.style.display === 'none' ? 'block' : 'none';
        });
        controlsDiv.insertBefore(advancedBtn, controlsDiv.firstChild);
    }
}

// Export state.graphData as a JSON file download
export function exportGraphData() {
    if (!state.graphData) return;

    try {
        const dataStr = JSON.stringify(state.graphData, null, 2);
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);

        const exportFileDefaultName = `knowledge_graph_${new Date().toISOString().split('T')[0]}.json`;

        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.click();

        showSuccess('Graph exported successfully!');
    } catch (error) {
        showError('Failed to export graph data');
    }
}

// Resize nodes based on selected metric (fixed, degree, importance)
function handleNodeSizeChange() {
    state.nodeSizeMetric = document.getElementById('node-size-metric').value;

    if (!state.network || !state.graphData) return;

    const nodes = state.network.body.data.nodes;
    state.graphData.nodes.forEach(node => {
        let size = 30; // Default size

        if (state.nodeSizeMetric === 'degree') {
            // Size based on connection count
            const nodeConnections = state.graphData.relationships.filter(rel =>
                rel.from === node.id || rel.to === node.id
            ).length;
            size = Math.max(15, Math.min(50, 30 + nodeConnections * 3));
        } else if (state.nodeSizeMetric === 'importance') {
            // Size based on some importance metric (placeholder)
            // This could be improved with actual centrality calculations
            size = 30 + (parseInt(node.id) % 10);
        }

        nodes.update({ id: node.id, size });
    });
}

// Enable or disable vis-network physics simulation
function handlePhysicsToggle() {
    state.physicsEnabled = document.getElementById('physics-enabled').checked;
    if (state.network) {
        state.network.setOptions({ physics: { enabled: state.physicsEnabled } });
    }
}

// Placeholder for future shortest-path selection UX
function findShortestPath() {
    if (!state.network) return;

    // This is a placeholder - in a real implementation you'd:
    // 1. Let user select two nodes
    // 2. Calculate shortest path using graph algorithms
    // 3. Highlight the path

    showError('Select two nodes to find the shortest path between them');
}

// Zoom in by 20% with animation
function handleZoomIn() {
    if (!state.network) {
        showError('Please load a knowledge graph first');
        return;
    }
    try {
        state.network.moveTo({
            scale: state.network.getScale() * 1.2,
            animation: true
        });
    } catch (error) {
        console.error('Zoom in error:', error);
        showError('Failed to zoom in');
    }
}

// Zoom out by 20% with animation
function handleZoomOut() {
    if (!state.network) {
        showError('Please load a knowledge graph first');
        return;
    }
    try {
        state.network.moveTo({
            scale: state.network.getScale() * 0.8,
            animation: true
        });
    } catch (error) {
        console.error('Zoom out error:', error);
        showError('Failed to zoom out');
    }
}

// Reset filters, highlights, and view position to initial state
function handleResetZoom() {
    if (!state.network) {
        showError('Please load a knowledge graph first');
        return;
    }

    try {
        // Reset filters and highlights
        state.currentFilters.nodeTypes.clear();
        state.currentFilters.relationshipTypes.clear();
        state.highlightedNodes.clear();

        // Reset checkboxes to checked state
        const nodeCheckboxes = document.querySelectorAll('#node-type-filters input[type="checkbox"]');
        const relCheckboxes = document.querySelectorAll('#relationship-type-filters input[type="checkbox"]');
        nodeCheckboxes.forEach(checkbox => checkbox.checked = true);
        relCheckboxes.forEach(checkbox => checkbox.checked = true);

        // Clear RAG-specific filtering by reinitializing with full graph data
        if (state.graphData) {
            initializeGraph(state.graphData);
        }

        // Then reset zoom/view position
        // Small delay to ensure graph re-initialization is complete
        setTimeout(() => {
            try {
                // First try to restore to initial state if we have it
                if (state.initialViewState && state.initialViewState.position && state.initialViewState.scale) {
                    // Try to restore to the saved initial view state
                    state.network.moveTo({
                        position: state.initialViewState.position,
                        scale: state.initialViewState.scale,
                        animation: true
                    });
                } else {
                    // Fallback: Fit the entire graph (similar to Neo4j's reset zoom behavior)
                    state.network.fit({
                        animation: true
                    });
                }
            } catch (error) {
                console.error('Reset zoom error:', error);
                showError('Failed to reset view');
            }
        }, 100);
    } catch (error) {
        console.error('Reset error:', error);
        showError('Failed to reset graph state');
    }
}

// Wire up all graph control event listeners
export function initControls() {
    document.getElementById('physics-enabled').addEventListener('change', handlePhysicsToggle);
    document.getElementById('node-size-metric').addEventListener('change', handleNodeSizeChange);

    document.getElementById('zoom-in').addEventListener('click', handleZoomIn);
    document.getElementById('zoom-out').addEventListener('click', handleZoomOut);
    document.getElementById('reset-zoom').addEventListener('click', handleResetZoom);

    // Clear all highlighted nodes and refresh graph styles
    document.getElementById('clear-hl-btn').addEventListener('click', function() {
        state.highlightedNodes.clear();
        updateHighlightBadge(0);
        if (state.network) {
            applyHighlightStyles();
        } else if (state.graphData) {
            initializeGraph(state.graphData);
        }
    });

    // Export canvas as PNG
    document.getElementById('export-png').addEventListener('click', function() {
        if (!state.network) { showError('Please load a knowledge graph first'); return; }
        try {
            const canvas = state.network.canvas.frame.canvas;
            const link = document.createElement('a');
            link.download = `kg_${state.currentKGName || 'graph'}_${new Date().toISOString().slice(0,10)}.png`;
            link.href = canvas.toDataURL('image/png');
            link.click();
        } catch (error) {
            showError('PNG export failed: ' + error.message);
        }
    });

    // Export JSON (SVG not feasible without extra libs — export graph data as JSON instead)
    document.getElementById('export-svg').addEventListener('click', function() {
        if (!state.graphData) { showError('Please load a knowledge graph first'); return; }
        exportGraphData();
    });

    // Close node detail panel
    document.getElementById('node-panel-close').addEventListener('click', function() {
        document.getElementById('node-detail-panel').classList.remove('open');
    });
}
