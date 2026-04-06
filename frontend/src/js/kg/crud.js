import { state } from '../state.js';
import { api } from '../api.js';
import { showError, showSuccess } from '../ui/notifications.js';
import { addToChat } from '../chat/messages.js';
import { initializeGraph } from '../graph/network.js';
import {
  updateKGBadge,
  updateChatKGName,
  updateHighlightBadge,
} from '../graph/legend.js';

// Store KG ID for later use with Load from Neo4j button
function storeKGId(kgId) {
  localStorage.setItem('currentKGId', kgId);
  console.log('Stored KG ID:', kgId);
}

// KG build progress streaming
let _progressSource = null;

function startProgressStream() {
  const panel = document.getElementById('kg-progress-panel');
  const log = document.getElementById('kg-progress-log');
  if (!panel || !log) return;
  log.innerHTML = '';
  panel.style.opacity = '';
  panel.style.display = 'block';

  if (_progressSource) { _progressSource.close(); _progressSource = null; }
  _progressSource = new EventSource('/kg_progress_stream');

  _progressSource.onmessage = function (e) {
    try {
      const data = JSON.parse(e.data);
      if (data.done) {
        appendLog(log, '✓ Done', 'pl-done');
        _progressSource.close(); _progressSource = null;
        // Auto-hide after 3 s with a fade-out
        setTimeout(() => {
          panel.style.opacity = '0';
          setTimeout(() => { panel.style.display = 'none'; panel.style.opacity = ''; }, 400);
        }, 3000);
        return;
      }
      if (data.line) {
        const cls = data.line.startsWith('❌') || data.line.startsWith('Error') ? 'pl-err'
          : data.line.startsWith('✓') || data.line.startsWith('🎉') ? 'pl-done'
          : data.line.startsWith('🔍') || data.line.startsWith('📊') ? 'pl-step'
          : '';
        appendLog(log, data.line, cls);
      }
    } catch (err) { /* ignore parse errors */ }
  };
  _progressSource.onerror = function () {
    if (_progressSource) { _progressSource.close(); _progressSource = null; }
  };
}

function stopProgressStream() {
  if (_progressSource) { _progressSource.close(); _progressSource = null; }
}

function appendLog(logEl, text, cls) {
  const p = document.createElement('p');
  p.className = 'pl' + (cls ? ' ' + cls : '');
  p.textContent = text;
  logEl.appendChild(p);
  logEl.scrollTop = logEl.scrollHeight;
}

async function loadKGList() {
  try {
    const data = await api.fetchKGList();
    const kgNameCreate = document.getElementById('kg-name-create');
    const kgLabelFilter = document.getElementById('kg-label-filter');

    // Clear the creation dropdown (keep first option)
    while (kgNameCreate.options.length > 1) {
      kgNameCreate.remove(1);
    }

    // Clear the filter dropdown in neo4j form (keep first option)
    while (kgLabelFilter.options.length > 1) {
      kgLabelFilter.remove(1);
    }

    // Add KGs from the response to all dropdowns
    if (data.kgs && data.kgs.length > 0) {
      data.kgs.forEach(kg => {
        // Add to creation dropdown
        const option2 = document.createElement('option');
        option2.value = kg.name;
        option2.textContent = kg.name;
        kgNameCreate.appendChild(option2);

        // Add to filter dropdown in neo4j form
        const option3 = document.createElement('option');
        option3.value = kg.name;
        option3.textContent = kg.name;
        kgLabelFilter.appendChild(option3);
      });
    }
  } catch (error) {
    console.error('Error loading KG list:', error);
  }
}

async function handleCreateKG() {
  const createButton = document.getElementById('create-kg-btn');
  const originalText = createButton.textContent;

  try {
    // Show loading spinner
    createButton.innerHTML = '<div class="spinner"></div> Creating KG...';
    createButton.disabled = true;

    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];
    if (!file) return;

    const ontologyFileInput = document.getElementById('ontology-upload');
    const ontologyFile = ontologyFileInput.files[0];

    const provider = document.getElementById('kg-provider').value;
    const model = document.getElementById('kg-model').value;
    const embeddingModel = document.getElementById('embedding-model').value;
    const maxChunks = document.getElementById('max-chunks').value;

    // Get KG name - either from dropdown (existing KG) or input (new KG)
    const kgNameSelect = document.getElementById('kg-name-create');
    const kgNameInput = document.getElementById('kg-name-new');
    let kgName = kgNameSelect.value;

    // If "New KG" selected, use the input value
    if (!kgName && kgNameInput.value.trim()) {
      kgName = kgNameInput.value.trim();
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('provider', provider);
    formData.append('model', model);
    formData.append('embedding_model', embeddingModel);
    formData.append('max_chunks', maxChunks);
    if (kgName) {
      formData.append('kg_name', kgName);
    }
    if (ontologyFile) {
      formData.append('ontology_file', ontologyFile);
    }

    // Start progress stream before the fetch
    startProgressStream();
    const result = await api.createKG(formData);
    stopProgressStream();

    state.currentKGId = result.kg_id;
    state.currentKGName = result.kg_name || null;
    if (state.currentKGName) localStorage.setItem('currentKGName', state.currentKGName);
    updateKGBadge(state.currentKGName);
    updateChatKGName(state.currentKGName);
    const sbCreate = document.getElementById('sample-badge');
    if (sbCreate) sbCreate.style.display = 'none';
    console.log('KG creation result:', result);
    console.log('Graph data:', result.graph_data);

    // Display success message (graph will be loaded via Load from Neo4j button)
    const nodeCount = result.graph_data?.nodes?.length ?? 0;
    const relCount = result.graph_data?.relationships?.length ?? 0;
    // Update progress panel with final success
    const plog = document.getElementById('kg-progress-log');
    if (plog) {
      const p = document.createElement('p');
      p.className = 'pl pl-done';
      p.textContent = `🎉 Done — ${nodeCount} nodes, ${relCount} edges`;
      plog.appendChild(p);
      plog.scrollTop = plog.scrollHeight;
    }
    addToChat(`Knowledge graph created with ${nodeCount} nodes and ${relCount} relationships.`);

    // Store the KG ID for later use with the Load from Neo4j button
    storeKGId(result.kg_id);

    // Auto-load the graph immediately
    if (result.graph_data) {
      state.graphData = result.graph_data;
      initializeGraph(state.graphData);
    }

    // Reset file selection
    fileInput.value = '';
    document.getElementById('file-selection').style.display = 'none';
  } catch (error) {
    stopProgressStream();
    showError(`KG creation failed: ${error.message}`);
  } finally {
    // Restore button state
    createButton.innerHTML = originalText;
    createButton.disabled = false;
  }
}

async function handleConnectNeo4j() {
  const connectButton = document.getElementById('connect-neo4j');
  const originalText = connectButton.textContent;

  try {
    // Show loading spinner
    connectButton.innerHTML = '<div class="spinner"></div> Loading...';
    connectButton.disabled = true;

    const uri = document.getElementById('neo4j-uri').value;
    const user = document.getElementById('neo4j-user').value;
    // For demo access, password is not required
    const passwordElement = document.getElementById('neo4j-password');
    const password = passwordElement.style.display === 'none' ? '' : passwordElement.value;

    if (!uri || !user || (passwordElement.style.display !== 'none' && !password)) {
      showError('Please fill in all required fields');
      return;
    }

    // Get selected load mode
    const loadMode = document.querySelector('input[name="load-mode"]:checked').value;
    const nodeLimit = document.getElementById('node-limit').value;
    const kgLabelFilter = document.getElementById('kg-label-filter').value.trim();

    const formData = new FormData();
    formData.append('uri', uri);
    formData.append('user', user);
    formData.append('password', password);

    // Add optional parameters based on selection
    if (kgLabelFilter) {
      formData.append('kg_label', kgLabelFilter);
    }

    // Set parameters based on load mode
    switch (loadMode) {
      case 'limited':
        formData.append('limit', nodeLimit);
        formData.append('sample_mode', 'false');
        formData.append('load_complete', 'false');
        break;
      case 'sample':
        formData.append('sample_mode', 'true');
        formData.append('load_complete', 'false');
        if (nodeLimit) {
          formData.append('limit', nodeLimit);
        }
        break;
      case 'complete':
        formData.append('load_complete', 'true');
        formData.append('sample_mode', 'false');
        break;
    }

    const result = await api.loadFromNeo4j(formData);

    state.currentKGId = result.kg_id;
    state.currentKGName = result.kg_name || kgLabelFilter || null;
    if (state.currentKGName) localStorage.setItem('currentKGName', state.currentKGName);
    updateKGBadge(state.currentKGName);
    updateChatKGName(state.currentKGName);
    initializeGraph(result.graph_data);
    state.graphData = result.graph_data; // Store graph data for re-initialization
    state.highlightedNodes.clear();
    updateHighlightBadge(0);
    // Clear filters for new graph
    state.currentFilters.nodeTypes.clear();
    state.currentFilters.relationshipTypes.clear();

    // Display detailed loading statistics
    const stats = result.stats || {};

    // Show sample-mode badge only when explicitly sampling
    const sampleBadge = document.getElementById('sample-badge');
    if (sampleBadge) {
      if (stats.sample_mode) {
        sampleBadge.textContent = `⚠ partial graph (${stats.loaded_nodes || 0} / ${stats.total_nodes_in_db || '?'} nodes)`;
        sampleBadge.style.display = 'block';
      } else {
        sampleBadge.style.display = 'none';
      }
    }
    const statsDiv = document.getElementById('neo4j-stats');
    document.getElementById('db-node-count').textContent = stats.total_nodes_in_db || '-';
    document.getElementById('db-rel-count').textContent = stats.total_relationships_in_db || '-';
    document.getElementById('loaded-node-count').textContent = stats.loaded_nodes || '-';
    document.getElementById('loaded-rel-count').textContent = stats.loaded_relationships || '-';
    statsDiv.style.display = 'block';

    // Show success message with details
    let nodeCount = result.graph_data?.nodes?.length ?? 0;
    let relCount = result.graph_data?.relationships?.length ?? 0;
    let successMsg = result.message || `Loaded KG from Neo4j with ${nodeCount} nodes and ${relCount} relationships`;
    if (stats.sample_mode) {
      successMsg += ' (Smart Sample)';
    } else if (stats.complete_import) {
      successMsg += ' (Complete Import)';
    }

    addToChat(successMsg, 'ai');
    showError(successMsg);

    // Hide form after successful connection
    document.getElementById('neo4j-form').style.display = 'none';
  } catch (error) {
    // Handle different error types
    let errorMsg;
    if (error instanceof Error) {
      errorMsg = error.message;
    } else if (typeof error === 'object') {
      try {
        errorMsg = JSON.stringify(error);
      } catch (e) {
        errorMsg = 'Unknown error object';
      }
    } else {
      errorMsg = String(error);
    }
    showError(`Neo4j loading failed: ${errorMsg}`);
  } finally {
    // Restore button state
    connectButton.innerHTML = originalText;
    connectButton.disabled = false;
  }
}

export function initKG() {
  // Pre-fill non-sensitive Neo4j connection defaults
  api.fetchDefaultCredentials()
    .then(data => {
      if (data.uri) document.getElementById('neo4j-uri').value = data.uri;
      if (data.user) document.getElementById('neo4j-user').value = data.user;
      if (data.uri) document.getElementById('save-neo4j-uri').value = data.uri;
      if (data.user) document.getElementById('save-neo4j-user').value = data.user;
      // Password is never sent by the server — user must type it
    })
    .catch(() => { /* non-fatal — form remains empty for user to fill */ });

  // Load KG list on page load
  loadKGList();

  // KG name create dropdown change handler
  document.getElementById('kg-name-create').addEventListener('change', function () {
    const select = this;
    const input = document.getElementById('kg-name-new');

    if (select.value === '') {
      // "New KG" selected - show input
      input.style.display = 'block';
      input.required = true;
    } else {
      // Existing KG selected - hide input
      input.style.display = 'none';
      input.required = false;
      input.value = '';
    }
  });

  // File upload triggers
  document.getElementById('load-file').addEventListener('click', function () {
    document.getElementById('file-upload').click();
  });

  document.getElementById('file-upload').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) return;

    // Show selected file and create button
    document.getElementById('selected-file').textContent = `Selected: ${file.name}`;
    document.getElementById('file-selection').style.display = 'block';
  });

  // Ontology upload
  document.getElementById('select-ontology').addEventListener('click', function () {
    document.getElementById('ontology-upload').click();
  });

  document.getElementById('ontology-upload').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file) {
      if (!file.name.endsWith('.json') && !file.name.endsWith('.owl')) {
        showError('Only JSON and OWL ontology files are supported');
        e.target.value = '';
        document.getElementById('selected-ontology').textContent = '';
        return;
      }
      document.getElementById('selected-ontology').textContent = `Ontology: ${file.name}`;
    }
  });

  // Neo4j form open/close
  document.getElementById('load-neo4j').addEventListener('click', function () {
    // Refresh KG list when opening the form
    loadKGList();
    document.getElementById('neo4j-form').style.display = 'block';
  });

  document.getElementById('close-neo4j-form').addEventListener('click', function () {
    document.getElementById('neo4j-form').style.display = 'none';
  });

  document.getElementById('close-save-neo4j-form').addEventListener('click', function () {
    document.getElementById('save-neo4j-form').style.display = 'none';
  });

  // Connect neo4j button
  document.getElementById('connect-neo4j').addEventListener('click', handleConnectNeo4j);

  // Create KG handler
  document.getElementById('create-kg-btn').addEventListener('click', handleCreateKG);

  // Progress panel close button
  document.getElementById('progress-panel-close').addEventListener('click', function () {
    document.getElementById('kg-progress-panel').style.display = 'none';
  });

  // Clear KG button functionality
  document.getElementById('clear-kg-btn').addEventListener('click', async function () {
    if (!confirm('⚠️ WARNING: This will permanently delete ALL nodes and relationships from Neo4j database. This action cannot be undone. Are you sure you want to continue?')) {
      return;
    }

    const clearButton = document.getElementById('clear-kg-btn');
    const originalText = clearButton.textContent;

    try {
      // Show loading state
      clearButton.innerHTML = '<div class="spinner"></div> Clearing...';
      clearButton.disabled = true;

      const result = await api.clearKG();

      // Clear the current graph visualization
      if (state.network) {
        try {
          state.network.destroy();
          state.network = null;
        } catch (e) {
          console.warn('Error destroying network:', e);
        }
      }

      // Clear stored graph data
      state.graphData = null;
      state.currentKGId = null;
      state.currentKGName = null;
      localStorage.removeItem('currentKGName');
      updateKGBadge(null);
      updateChatKGName(null);
      updateHighlightBadge(0);
      state.highlightedNodes.clear();
      const sb = document.getElementById('sample-badge');
      if (sb) sb.style.display = 'none';

      // Clear overview panel
      document.getElementById('overview-panel').style.display = 'none';

      // Refresh KG list so dropdown shows empty state
      await loadKGList();

      // Add success message to chat
      addToChat(`🧹 ${result.message}`, 'ai');

      // Show success notification
      showError(`✅ Knowledge graph cleared successfully!`);
    } catch (error) {
      console.error('Error clearing KG:', error);
      showError(`❌ Failed to clear knowledge graph: ${error.message}`);
      addToChat(`Error clearing KG: ${error.message}`, 'error');
    } finally {
      // Restore button state
      clearButton.innerHTML = originalText;
      clearButton.disabled = false;
    }
  });
}
