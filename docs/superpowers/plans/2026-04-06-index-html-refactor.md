# index.html Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the monolithic 4,809-line `ontographrag/api/static/index.html` into a Vite-based frontend project with ES modules and separated CSS files.

**Architecture:** Create a `frontend/` directory at project root with Vite, split inline CSS into 6 concern-based files, split inline JS into 15 feature-based ES modules sharing a single state object. FastAPI serves the built output from `frontend/dist/`.

**Tech Stack:** Vite 6, ES modules, vis-network (CDN), Prism.js (CDN), Font Awesome (CDN), FastAPI (serving built assets)

**Source file:** `ontographrag/api/static/index.html` — all line references below refer to this file.

---

### Task 1: Scaffold the frontend project

**Files:**

- Create: `frontend/package.json`
- Create: `frontend/vite.config.js`
- Create: `frontend/src/main.js` (placeholder)

- [ ] **Step 1: Create `frontend/package.json`**

```json
{
  "name": "medigraphrag-frontend",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "devDependencies": {
    "vite": "^6"
  }
}
```

- [ ] **Step 2: Create `frontend/vite.config.js`**

```js
import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/models': 'http://localhost:8000',
      '/kg': 'http://localhost:8000',
      '/chat': 'http://localhost:8000',
      '/doctor': 'http://localhost:8000',
      '/neo4j': 'http://localhost:8000',
      '/create_ontology_guided_kg': 'http://localhost:8000',
      '/load_kg_from_neo4j': 'http://localhost:8000',
      '/clear_kg': 'http://localhost:8000',
      '/kg_progress_stream': 'http://localhost:8000',
      '/static': 'http://localhost:8000',
    },
  },
});
```

- [ ] **Step 3: Create placeholder `frontend/src/main.js`**

```js
// Entry point — will import CSS and init modules after extraction
console.log('frontend loaded');
```

- [ ] **Step 4: Add frontend entries to `.gitignore`**

Add these lines to the project root `.gitignore`:

```text
# Frontend
frontend/node_modules/
frontend/dist/
```

- [ ] **Step 5: Install dependencies and verify build**

```bash
cd frontend && npm install && npm run build
```

Expected: Build succeeds, `frontend/dist/` directory created.

- [ ] **Step 6: Commit**

```bash
git add frontend/package.json frontend/vite.config.js frontend/src/main.js frontend/package-lock.json .gitignore
git commit -m "feat: scaffold vite frontend project"
```

---

### Task 2: Extract HTML into `frontend/index.html`

**Files:**

- Create: `frontend/index.html`
- Source: `ontographrag/api/static/index.html` lines 1-11 (head/meta/CDN links), lines 1912-2246 (body markup)

- [ ] **Step 1: Create `frontend/index.html`**

Extract the HTML structure from the source file. The file should contain:

1. Lines 1-11: `<!DOCTYPE html>` through the CDN `<link>` tags (vis-network CSS, Prism CSS, Font Awesome CSS)
2. Remove the opening `<style>` tag (line 12) and everything through `</style>` (line 1910)
3. Keep `</head>` (line 1911)
4. Lines 1912-2246: The full `<body>` markup from `<body data-theme="dark">` through the closing `</div>` of save-neo4j-form
5. Remove the `<script src="...vis-network...">` tag (line 2247) — vis-network will still be loaded via CDN but as a separate script tag before the module
6. Remove the entire inline `<script>` block (lines 2248-4807)
7. Add vis-network CDN script and the Vite module entry point before `</body>`

The result should look like this (abbreviated — the full body markup from lines 1912-2246 goes in the middle):

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph RAG - Professional KG Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/vis-network@9.1.0/dist/vis-network.min.css" rel="stylesheet" type="text/css" />
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" rel="stylesheet" />
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet" />
</head>
<body data-theme="dark">
    <!-- ... all body markup from lines 1912-2246 verbatim ... -->

    <script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.0/dist/vis-network.min.js"></script>
    <script type="module" src="/src/main.js"></script>
</body>
</html>
```

Copy lines 1912-2246 from the source file exactly as-is for the body content. Do not modify any IDs, classes, or structure.

- [ ] **Step 2: Verify build**

```bash
cd frontend && npm run build
```

Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add frontend/index.html
git commit -m "feat: extract HTML markup into frontend/index.html"
```

---

### Task 3: Extract CSS — `base.css` and `animations.css`

**Files:**

- Create: `frontend/src/css/base.css`
- Create: `frontend/src/css/animations.css`

- [ ] **Step 1: Create `frontend/src/css/base.css`**

Extract these sections from the source `<style>` block:

1. The `.hidden` rule (lines 72-74)
2. The `*` reset (lines 76-81)
3. The `body` rule (lines 83-89)
4. The `[data-theme="dark"]` variable block — look for the comment `/* ── Neutral Dark ──` (line 1402) through the end of its closing brace
5. The `[data-theme="light"]` variable block — look for the comment `/* ── Minimal Light ──` (line 1419) through the end of its closing brace
6. The `/* Base */` section (around line 1436)
7. The font floor section `/* ── Font floor: raise minimum to 12px ──` (line 1238)
8. The custom scrollbars section `/* ── 1. Custom scrollbars ──` (line 1853) through related scrollbar rules

Copy each CSS block exactly as-is. Remove the 8-space indentation prefix that exists because they were inside `<style>` in an HTML file.

- [ ] **Step 2: Create `frontend/src/css/animations.css`**

Extract these sections:

1. The `.spinner` rule (lines 49-59)
2. The `@keyframes spin` rule (lines 61-63)
3. The `.btn-dots` rules (lines 66-70)
4. The `@keyframes dotPulse` rule (line 70)

Remove the 8-space indentation prefix.

- [ ] **Step 3: Add CSS imports to `main.js`**

Update `frontend/src/main.js`:

```js
import './css/base.css';
import './css/animations.css';

console.log('frontend loaded');
```

- [ ] **Step 4: Verify build**

```bash
cd frontend && npm run build
```

Expected: Build succeeds.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/css/base.css frontend/src/css/animations.css frontend/src/main.js
git commit -m "feat: extract base and animation CSS"
```

---

### Task 4: Extract CSS — `layout.css`

**Files:**

- Create: `frontend/src/css/layout.css`

- [ ] **Step 1: Create `frontend/src/css/layout.css`**

Extract these sections from the source `<style>` block:

1. `/* Left Panel - Minimal Controls */` — `.left-panel` and its children (lines 91-100, continuing through `.panel-section`, `.control-group`, `.action-buttons`, `.btn-primary`, `.btn-secondary`, `.temp-display` — lines 91-190)
2. `/* Main Content Area */` — `.main-content` (lines 193-201)
3. `/* View Container */` — `.view-container` (lines 203-209)
4. `/* KG View */` — `#kg-view` and its h2 rule (lines 211-228)
5. `/* KG expand button */` — `.kg-expand-btn` (lines 494-514)
6. `/* KG expanded state */` — `#kg-view.kg-expanded`, `#chat-view.kg-expanded-hidden` (lines 516-522)
7. `/* Chat View */` — `#chat-view`, `.chat-container`, `#chat-section`, `#chat-box`, `.chat-input-container`, `#question`, `#send-btn` (lines 524-570)
8. `/* ── Collapsible sidebar ──` (line 1252) through its related rules
9. `/* Sidebar */` section (around line 1441)
10. `/* Main panels */` section (around line 1468)
11. `/* Responsive adjustments */` (line 1217) through related media queries

Remove 8-space indentation prefix. Some sections have 4-space indentation — normalize all to no leading whitespace or consistent 4-space nesting for nested selectors.

- [ ] **Step 2: Add CSS import to `main.js`**

Add after existing imports:

```js
import './css/layout.css';
```

- [ ] **Step 3: Verify build**

```bash
cd frontend && npm run build
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/css/layout.css frontend/src/main.js
git commit -m "feat: extract layout CSS"
```

---

### Task 5: Extract CSS — `graph.css`

**Files:**

- Create: `frontend/src/css/graph.css`

- [ ] **Step 1: Create `frontend/src/css/graph.css`**

Extract these sections from the source:

1. `/* Neo4j Overview Panel */` — `.overview-panel` and all `.overview-*` rules (lines 230-345)
2. `.graph-tools`, `.search-container`, `#node-search`, `.search-icon` (lines 347-373)
3. `.filter-btn` (lines 375-387)
4. `#graph-container` (lines 389-395)
5. `/* Filter Panel */` — `.filter-panel` and all `.filter-*` rules (lines 397-470)
6. `.graph-controls`, `.zoom-btn` (lines 472-492)
7. `/* ── Graph canvas — theme-aware ──` (line 1247)
8. `/* ── Active KG badge ──` (line 1294)
9. `/* ── Highlight mode badge ──` (line 1312)
10. `/* ── Mini-legend ──` (line 1358)
11. `/* Graph header */` (around line 1489)
12. `/* Graph controls strip */` (around line 1476)
13. `/* Filter panel */` theme overrides (around line 1502)
14. `/* Graph control selects */` (around line 1646)
15. `/* Overview panel */` theme rules (around line 1687)
16. `/* ── Empty graph state ──` (line 1758)
17. `/* ── Node detail side panel ──` (line 1767)
18. `/* Advanced controls */` (around line 1668)
19. `/* Buttons */` section (around line 1456) — only graph-related button overrides
20. Search clear button and search count styles (if present)
21. Sample badge styles (if present)

- [ ] **Step 2: Add CSS import to `main.js`**

```js
import './css/graph.css';
```

- [ ] **Step 3: Verify build**

```bash
cd frontend && npm run build
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/css/graph.css frontend/src/main.js
git commit -m "feat: extract graph CSS"
```

---

### Task 6: Extract CSS — `chat.css`

**Files:**

- Create: `frontend/src/css/chat.css`

- [ ] **Step 1: Create `frontend/src/css/chat.css`**

Extract these sections:

1. `/* Enhanced Clinical-Focused Message Styling */` — `.chat-messages`, `.message-container`, `.ai-message`, `.user-message` (lines 648-699)
2. `/* Clinical Context Indicators */` (line 707)
3. `/* Confidence Level Indicators */` (line 722)
4. `/* Enhanced Hover Interactions */` (line 754)
5. `/* Graph-Chat Synchronization */` (line 772)
6. `/* Timestamp styling */` (line 802)
7. `/* Enhanced Typography with Markdown Support */` (line 812)
8. `/* Enhanced Lists */` (line 861)
9. `/* Enhanced Code Blocks */` (line 901)
10. `/* Enhanced Blockquotes */` (line 932)
11. `/* Enhanced Text Formatting */` (line 944)
12. `/* Enhanced Links */` (line 962)
13. `/* Enhanced Tables */` (line 975)
14. `/* Clinical Response Styling - Enhanced */` (line 1003)
15. `/* Enhanced Icons */` (line 1093)
16. `/* Horizontal Rules */` (line 1100)
17. `/* Enhanced Paragraphs */` (line 1108)
18. `/* Section Toggle Styling */` (line 1122)
19. `/* Reasoning Steps Styling */` (line 1170)
20. `/* Clear chat button */` (line 1199)
21. `/* ── Chat redesign ──` (line 1512) through all chat-related rules in that block
22. `/* Section toggles in chat */` (line 1594)
23. `/* Thinking message */` (line 1608)
24. `/* Copy button on AI messages */` (line 1635)
25. `/* ── Source chip on AI responses ──` (line 1793)
26. `/* ── Sources panel ──` (line 1803)
27. `/* ── Suggested questions ──` (line 1815)
28. `/* ── 7. Keyboard hint below input ──` (line 1900)
29. `/* ── Light-theme specific adjustments ──` (line 1720) — only chat-related overrides
30. `/* ── Dark theme: error bar only ──` (line 1744) — chat-related parts

- [ ] **Step 2: Add CSS import to `main.js`**

```js
import './css/chat.css';
```

- [ ] **Step 3: Verify build**

```bash
cd frontend && npm run build
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/css/chat.css frontend/src/main.js
git commit -m "feat: extract chat CSS"
```

---

### Task 7: Extract CSS — `components.css`

**Files:**

- Create: `frontend/src/css/components.css`

- [ ] **Step 1: Create `frontend/src/css/components.css`**

Extract all remaining CSS rules not captured by previous tasks:

1. `/* Error Popup */` — `.error-popup`, `.error-content`, `#close-error-popup` (lines 13-46)
2. `/* File Selection */` — `#file-selection`, `#selected-file`, `#selected-ontology` (lines 572-591)
3. `/* Neo4j Forms */` — `#neo4j-form`, `#save-neo4j-form`, `.close-btn` (lines 593-646)
4. `/* Neo4j stats */` (line 1210)
5. `/* Neo4j forms */` theme overrides (line 1615)
6. `/* Error/notification bar */` (line 1629)
7. `/* Success popup variant */` (line 1654)
8. `/* Node info overlay — dark theme */` (line 1659)
9. `/* ── KG build progress panel ──` (line 1830)
10. `/* ── Theme toggle ──` (line 1694)
11. `/* Sidebar toggle button */` (line 1678)
12. Any remaining rules from `/* Buttons */` section (line 1456) not in graph.css
13. Any remaining radio button, select, input styles not in other files

- [ ] **Step 2: Add CSS import to `main.js`**

```js
import './css/components.css';
```

- [ ] **Step 3: Verify build**

```bash
cd frontend && npm run build
```

- [ ] **Step 4: Verify no CSS rules were missed**

Compare total CSS line count: the original `<style>` block spans lines 12-1910 (~1,898 lines). The sum of all 6 CSS files should be approximately equal (minus blank lines from deindenting).

Grep the original file for any CSS selectors and verify each appears in one of the 6 CSS files:

```bash
grep -oP '\.[a-z][\w-]+|#[a-z][\w-]+' ontographrag/api/static/index.html | sort -u | head -50
```

Spot-check that major selectors (`.left-panel`, `#graph-container`, `.ai-message`, `.error-popup`, `.spinner`) each appear in exactly one CSS file.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/css/components.css frontend/src/main.js
git commit -m "feat: extract components CSS"
```

---

### Task 8: Create JS leaf modules — `state.js`, `api.js`, `ui/notifications.js`

**Files:**

- Create: `frontend/src/js/state.js`
- Create: `frontend/src/js/api.js`
- Create: `frontend/src/js/ui/notifications.js`

These three modules have zero project imports — they are the foundation.

- [ ] **Step 1: Create `frontend/src/js/state.js`**

Extract global variables from lines 2249-2265:

```js
// Shared mutable application state.
// All modules import { state } and read/write properties directly.
export const state = {
  network: null,
  currentKGId: null,
  currentKGName: null,
  graphData: null,
  fullGraphData: null,
  highlightedNodes: new Set(),
  uniqueIdCounter: 0,
  nodeTypeColors: {},
  relationshipTypeColors: {},
  currentFilters: { nodeTypes: new Set(), relationshipTypes: new Set() },
  clusters: {},
  initialViewState: null,
  physicsEnabled: true,
  nodeSizeMetric: 'fixed',
  showEdgeLabels: true,
};

// Clear stale KG name from previous session on load
localStorage.removeItem('currentKGName');
```

- [ ] **Step 2: Create `frontend/src/js/api.js`**

```js
async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Request failed: ${url}`);
  }
  return response.json();
}

export const api = {
  fetchModels: (vendor) => request(`/models/${vendor}`),
  fetchKGList: () => request('/kg/list'),
  fetchDefaultCredentials: () => request('/neo4j/default_credentials'),
  checkHealth: () => request('/doctor'),
  clearKG: () => request('/clear_kg', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  }),
  createKG: (formData) => request('/create_ontology_guided_kg', {
    method: 'POST',
    body: formData,
  }),
  loadFromNeo4j: (formData) => request('/load_kg_from_neo4j', {
    method: 'POST',
    body: formData,
  }),
  sendChat: (payload, signal) => request('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  }),
};
```

- [ ] **Step 3: Create `frontend/src/js/ui/notifications.js`**

Extract `showError` (lines 3502-3506) and `showSuccess` (lines 3508-3514):

```js
export function showError(message) {
  const popup = document.getElementById('error-popup');
  document.getElementById('error-message').textContent = message;
  popup.classList.remove('hidden', 'is-success');
}

export function showSuccess(message) {
  const popup = document.getElementById('error-popup');
  document.getElementById('error-message').textContent = message;
  popup.classList.remove('hidden');
  popup.classList.add('is-success');
  setTimeout(() => popup.classList.add('hidden'), 3000);
}
```

- [ ] **Step 4: Verify build**

```bash
cd frontend && npm run build
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/js/state.js frontend/src/js/api.js frontend/src/js/ui/notifications.js
git commit -m "feat: create state, api, and notifications modules"
```

---

### Task 9: Create UI modules — `theme.js`, `sidebar.js`, `shortcuts.js`, `health.js`

**Files:**

- Create: `frontend/src/js/ui/theme.js`
- Create: `frontend/src/js/ui/sidebar.js`
- Create: `frontend/src/js/ui/shortcuts.js`
- Create: `frontend/src/js/ui/health.js`

- [ ] **Step 1: Create `frontend/src/js/ui/theme.js`**

Extract `getGraphTheme` (lines 2790-2801) and the theme toggle IIFE (lines 3832-3844):

```js
import { state } from '../state.js';

export function getGraphTheme() {
  const dark = document.body.dataset.theme === 'dark';
  return {
    nodeText:       dark ? '#ffffff' : '#1a1a1a',
    nodeTextDimmed: dark ? '#444444' : '#bbbbbb',
    edgeText:       dark ? '#888888' : '#555555',
    edgeLabelBg:    dark ? 'rgba(10,10,10,0.80)' : 'rgba(232,236,240,0.88)',
    dimmedNodeBg:   dark ? '#2a2a2a' : '#d8dde4',
    dimmedNodeBdr:  dark ? '#3a3a3a' : '#c8cdd4',
    dimmedEdge:     dark ? '#282828' : '#dde0e4',
  };
}

export function initTheme() {
  const saved = localStorage.getItem('kg-theme') || 'dark';
  document.body.dataset.theme = saved;
  document.getElementById('theme-icon').textContent = saved === 'dark' ? '🌙' : '☀️';

  document.getElementById('theme-toggle').addEventListener('click', () => {
    const next = document.body.dataset.theme === 'dark' ? 'light' : 'dark';
    document.body.dataset.theme = next;
    localStorage.setItem('kg-theme', next);
    document.getElementById('theme-icon').textContent = next === 'dark' ? '🌙' : '☀️';
    // Re-import initializeGraph lazily to avoid circular dependency
    if (state.graphData) {
      import('../graph/network.js').then(({ initializeGraph }) => {
        initializeGraph(state.graphData);
      });
    }
  });
}
```

Note: The theme toggle calls `initializeGraph(graphData)` when toggled (line 3842). To avoid a circular import (theme → network → theme), use a dynamic `import()` for this one call.

- [ ] **Step 2: Create `frontend/src/js/ui/sidebar.js`**

Extract sidebar collapse (lines 3770-3778) and KG expand (lines 3741-3767):

```js
import { state } from '../state.js';

export function initSidebar() {
  // Sidebar collapse toggle
  document.getElementById('sidebar-toggle').addEventListener('click', function () {
    const panel = document.querySelector('.left-panel');
    panel.classList.toggle('collapsed');
    const collapsed = panel.classList.contains('collapsed');
    this.textContent = collapsed ? '›' : '‹';
    this.title = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
    if (state.network) setTimeout(() => state.network.redraw(), 300);
  });

  // Default layout: both panels visible, KG at 62% / chat at 38%
  const kgView = document.getElementById('kg-view');
  const chatView = document.getElementById('chat-view');
  kgView.style.flex = '1.6';
  chatView.style.flex = '1';

  // KG expand / collapse toggle
  let kgExpanded = false;
  const expandBtn = document.getElementById('kg-expand-btn');
  const expandIconCollapse = `<svg viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" width="14" height="14"><path d="M6 1v5H1M15 6h-5V1M10 15v-5h5M1 10h5v5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  const expandIconExpand  = `<svg viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" width="14" height="14"><path d="M1 6V1h5M10 1h5v5M15 10v5h-5M6 15H1v-5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

  expandBtn.addEventListener('click', () => {
    kgExpanded = !kgExpanded;
    if (kgExpanded) {
      kgView.classList.add('kg-expanded');
      chatView.classList.add('kg-expanded-hidden');
      kgView.style.flex = '1';
      expandBtn.innerHTML = expandIconCollapse;
      expandBtn.title = 'Collapse graph view';
      expandBtn.setAttribute('aria-label', 'Collapse graph view');
    } else {
      kgView.classList.remove('kg-expanded');
      chatView.classList.remove('kg-expanded-hidden');
      kgView.style.flex = '1.6';
      chatView.style.flex = '1';
      expandBtn.innerHTML = expandIconExpand;
      expandBtn.title = 'Expand graph view';
      expandBtn.setAttribute('aria-label', 'Expand graph view');
    }
    if (state.network) setTimeout(() => state.network.redraw(), 150);
  });
}
```

- [ ] **Step 3: Create `frontend/src/js/ui/shortcuts.js`**

Extract keyboard shortcuts (lines 4436-4448, 4780-4787), `toggleSection` (lines 4790-4805), `toggleReasoningSteps` (lines 4741-4752), and `useSuggestion` (lines 3517-3520):

```js
import { performSearch } from '../graph/filters.js';

export function initShortcuts() {
  // Esc: clear node search + close node detail panel
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      const searchInput = document.getElementById('node-search');
      if (searchInput && searchInput.value) {
        searchInput.value = '';
        performSearch('');
      }
      const ndPanel = document.getElementById('node-detail-panel');
      if (ndPanel) ndPanel.classList.remove('open');
    }
  });

  // Cmd+K / Ctrl+K focuses the chat input
  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      const q = document.getElementById('question');
      if (q) { q.focus(); q.select(); }
    }
  });

  // Suggestion chips: fill textarea and focus
  window.useSuggestion = (btn) => {
    const q = document.getElementById('question');
    if (q) { q.value = btn.textContent.trim(); q.focus(); }
  };

  // Toggle response sections (used via onclick in HTML)
  window.toggleSection = (button) => {
    const sectionContent = button.nextElementSibling;
    const isExpanded = sectionContent.classList.contains('expanded');

    if (isExpanded) {
      sectionContent.classList.remove('expanded');
      sectionContent.classList.add('collapsed');
      button.textContent = button.textContent.replace('▼', '▶');
      button.classList.remove('expanded');
    } else {
      sectionContent.classList.remove('collapsed');
      sectionContent.classList.add('expanded');
      button.textContent = button.textContent.replace('▶', '▼');
      button.classList.add('expanded');
    }
  };

  // Toggle reasoning steps
  window.toggleReasoningSteps = () => {
    const reasoningStepsContent = document.getElementById('reasoning-steps-content');
    const toggleButton = document.querySelector('.reasoning-steps-toggle');

    if (reasoningStepsContent.style.display === 'none') {
      reasoningStepsContent.style.display = 'block';
      toggleButton.textContent = 'Hide Reasoning Steps';
    } else {
      reasoningStepsContent.style.display = 'none';
      toggleButton.textContent = 'Show Reasoning Steps';
    }
  };
}
```

Note: `shortcuts.js` imports `performSearch` from `graph/filters.js`. This is fine — `filters.js` does not import from `ui/`.

- [ ] **Step 4: Create `frontend/src/js/ui/health.js`**

Extract the doctor health check IIFE (lines 3781-3818):

```js
import { api } from '../api.js';

export function initHealth() {
  const dot = document.getElementById('doctor-dot');
  if (!dot) return;

  async function checkHealth() {
    try {
      const d = await api.checkHealth();
      const colors = { ok: '#2ecc71', warn: '#f39c12', fail: '#e74c3c' };
      dot.style.background = colors[d.status] || '#888';
      const failed = (d.checks || []).filter(c => c.status === 'fail').map(c => c.check).join(', ');
      const warned = (d.checks || []).filter(c => c.status === 'warn').map(c => c.check).join(', ');
      let tip = `System: ${d.status.toUpperCase()}`;
      if (failed) tip += `\nFailed: ${failed}`;
      if (warned) tip += `\nWarnings: ${warned}`;
      dot.title = tip;
    } catch (_e) {
      dot.style.background = '#e74c3c';
      dot.title = 'Health check failed — server may be unreachable';
    }
  }

  checkHealth();

  dot.addEventListener('click', async () => {
    dot.style.background = 'var(--text-3)';
    dot.title = 'Rechecking…';
    await checkHealth();
  });
}
```

- [ ] **Step 5: Verify build**

```bash
cd frontend && npm run build
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/js/ui/
git commit -m "feat: create UI modules (theme, sidebar, shortcuts, health)"
```

---

### Task 10: Create `graph/legend.js`

**Files:**

- Create: `frontend/src/js/graph/legend.js`

This module contains color generation, legend/overview rendering, badge updates, highlight styles, and confidence colors.

- [ ] **Step 1: Create `frontend/src/js/graph/legend.js`**

Extract these functions from the source:

- `generateNodeTypeColors` (lines 2357-2370)
- `generateRelationshipTypeColors` (lines 2373-2384)
- `updateLegend` (lines 2387-2438)
- `updateOverview` (lines 2441-2561) — note: this function calls `initializeGraph` when overview items are clicked. To avoid circular imports (legend → network → legend), use a dynamic `import()` for that call.
- `updateKGBadge` (lines 2805-2814)
- `updateChatKGName` (lines 2816-2825)
- `updateHighlightBadge` (lines 2827-2837)
- `updateMiniLegend` (lines 2839-2849)
- `confidenceEdgeColor` (lines 2852-2858)
- `applyHighlightStyles` (lines 2721-2787)

```js
import { state } from '../state.js';
import { getGraphTheme } from '../ui/theme.js';

export function generateNodeTypeColors(nodeTypes) {
  // ... extract from lines 2357-2370, replace return with assignment to state
}

// ... all other functions extracted verbatim

// For updateOverview, where it calls initializeGraph on click:
// Replace: initializeGraph(graphData)
// With: import('../graph/network.js').then(({ initializeGraph }) => initializeGraph(state.graphData))
```

Every function that previously read global variables like `nodeTypeColors`, `graphData`, `highlightedNodes`, etc. must now read from `state.nodeTypeColors`, `state.graphData`, `state.highlightedNodes`, etc.

Export all functions individually: `export function generateNodeTypeColors(...)`, etc.

- [ ] **Step 2: Verify build**

```bash
cd frontend && npm run build
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/js/graph/legend.js
git commit -m "feat: create graph legend module"
```

---

### Task 11: Create `graph/network.js`

**Files:**

- Create: `frontend/src/js/graph/network.js`

This is the largest and most critical module — the vis-network initialization.

- [ ] **Step 1: Create `frontend/src/js/graph/network.js`**

Extract `initializeGraph` (lines 2860-3415) from the source.

```js
import { state } from '../state.js';
import { getGraphTheme } from '../ui/theme.js';
import { showError } from '../ui/notifications.js';
import {
  generateNodeTypeColors,
  generateRelationshipTypeColors,
  updateLegend,
  updateOverview,
  updateMiniLegend,
  confidenceEdgeColor,
} from './legend.js';
import { updateFilterPanel } from './filters.js';

export function initializeGraph(data, mergeMode = false) {
  // ... extract function body from lines 2860-3415
  // Replace all bare variable references:
  //   network       → state.network
  //   graphData     → state.graphData
  //   nodeTypeColors → state.nodeTypeColors
  //   relationshipTypeColors → state.relationshipTypeColors
  //   currentFilters → state.currentFilters
  //   highlightedNodes → state.highlightedNodes
  //   uniqueIdCounter → state.uniqueIdCounter (use state.uniqueIdCounter++ for increments)
  //   initialViewState → state.initialViewState
  //   showEdgeLabels → state.showEdgeLabels
  //
  // The `data` parameter replaces the old `graphData` function parameter.
  // Where the function uses `graphData` as the parameter name internally,
  // keep using `data` to avoid confusion with `state.graphData`.
  //
  // vis.DataSet and vis.Network are globals from the CDN script — use them directly.
}
```

Key points for the extraction:

1. The function parameter is `graphData` but `state.graphData` is also used. Rename the parameter to `data` to avoid shadowing.
2. The `nodes` variable on line 3376 (`const node = nodes.get(nodeId)`) refers to the `vis.DataSet` created on line 3182 — this is a local variable inside the function, not a state variable.
3. The `network.on("click", ...)` handler (lines 3365-3414) references `nodes` — this local `vis.DataSet` variable must be in scope. It's defined inside the `else` branch (line 3182). The click handler is also inside that branch, so scoping is preserved.

- [ ] **Step 2: Verify build**

```bash
cd frontend && npm run build
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/js/graph/network.js
git commit -m "feat: create graph network module"
```

---

### Task 12: Create `graph/filters.js` and `graph/controls.js`

**Files:**

- Create: `frontend/src/js/graph/filters.js`
- Create: `frontend/src/js/graph/controls.js`

- [ ] **Step 1: Create `frontend/src/js/graph/filters.js`**

Extract these functions:

- `updateFilterPanel` (lines 2564-2638)
- `applyFilters` (lines 2641-2667)
- `performSearch` (lines 2670-2717)

```js
import { state } from '../state.js';
import { initializeGraph } from './network.js';

export function updateFilterPanel(nodeTypes, relationshipTypes) {
  // ... extract from lines 2564-2638
  // Replace graphData → state.graphData
}

export function applyFilters() {
  // ... extract from lines 2641-2667
  // Replace network → state.network, graphData → state.graphData
  // Replace initializeGraph(graphData) → initializeGraph(state.graphData)
}

export function performSearch(searchTerm) {
  // ... extract from lines 2670-2717
  // Replace network → state.network
}

export function initFilters() {
  // Wire up filter UI (extracted from DOMContentLoaded):
  // - Search input handler (line 4676-4678)
  // - Search clear button (lines 4679-4682)
  // - Filter panel toggle (lines 4685-4688)
  // - Apply filters button (lines 4691-4694)
  // - Reset filters button (lines 4697-4718)
  // - Edge labels toggle (lines 4451-4454)

  document.getElementById('node-search').addEventListener('input', (e) => {
    performSearch(e.target.value);
  });

  document.getElementById('search-clear').addEventListener('click', () => {
    document.getElementById('node-search').value = '';
    performSearch('');
  });

  document.getElementById('filter-btn').addEventListener('click', () => {
    const filterPanel = document.getElementById('filter-panel');
    filterPanel.style.display = filterPanel.style.display === 'none' ? 'block' : 'none';
  });

  document.getElementById('apply-filters').addEventListener('click', () => {
    applyFilters();
    document.getElementById('filter-panel').style.display = 'none';
  });

  document.getElementById('reset-filters').addEventListener('click', () => {
    const checkboxes = document.querySelectorAll('#filter-panel input[type="checkbox"]');
    checkboxes.forEach(checkbox => checkbox.checked = true);
    state.currentFilters.nodeTypes.clear();
    state.currentFilters.relationshipTypes.clear();
    if (state.graphData) initializeGraph(state.graphData);
    document.getElementById('filter-panel').style.display = 'none';
  });

  document.getElementById('edge-labels').addEventListener('change', function () {
    state.showEdgeLabels = this.checked;
    if (state.graphData) initializeGraph(state.graphData);
  });
}
```

- [ ] **Step 2: Create `frontend/src/js/graph/controls.js`**

Extract these functions:

- `createAdvancedButton` (lines 2268-2282)
- `handleNodeSizeChange` (lines 2286-2309)
- `handlePhysicsToggle` (lines 2312-2317)
- `findShortestPath` (lines 2320-2329)
- `exportGraphData` (lines 2332-2350)
- `handleZoomIn` (lines 4586-4600)
- `handleZoomOut` (lines 4602-4616)
- `handleResetZoom` (lines 4618-4668)
- Export PNG (lines 4721-4732)
- Export JSON/SVG (lines 4735-4738)

```js
import { state } from '../state.js';
import { showError, showSuccess } from '../ui/notifications.js';
import { initializeGraph } from './network.js';
import { applyHighlightStyles, updateHighlightBadge } from './legend.js';

export function createAdvancedButton() {
  // ... extract from lines 2268-2282
}

function handleNodeSizeChange() {
  // ... extract from lines 2286-2309
  // Replace nodeSizeMetric → state.nodeSizeMetric, network → state.network, graphData → state.graphData
}

function handlePhysicsToggle() {
  // ... extract from lines 2312-2317
  // Replace physicsEnabled → state.physicsEnabled, network → state.network
}

function findShortestPath() {
  // ... extract from lines 2320-2329
}

export function exportGraphData() {
  // ... extract from lines 2332-2350
  // Replace graphData → state.graphData
}

function handleZoomIn() {
  // ... extract from lines 4586-4600
}

function handleZoomOut() {
  // ... extract from lines 4602-4616
}

function handleResetZoom() {
  // ... extract from lines 4618-4668
  // Replace currentFilters → state.currentFilters, highlightedNodes → state.highlightedNodes,
  // graphData → state.graphData, initialViewState → state.initialViewState, network → state.network
}

export function initControls() {
  // Wire up graph control UI:
  document.getElementById('physics-enabled').addEventListener('change', handlePhysicsToggle);
  document.getElementById('node-size-metric').addEventListener('change', handleNodeSizeChange);
  document.getElementById('zoom-in').addEventListener('click', handleZoomIn);
  document.getElementById('zoom-out').addEventListener('click', handleZoomOut);
  document.getElementById('reset-zoom').addEventListener('click', handleResetZoom);

  // Highlight badge clear button (lines 3821-3829)
  document.getElementById('clear-hl-btn').addEventListener('click', () => {
    state.highlightedNodes.clear();
    updateHighlightBadge(0);
    if (state.network) {
      applyHighlightStyles();
    } else if (state.graphData) {
      initializeGraph(state.graphData);
    }
  });

  // Export PNG (lines 4721-4732)
  document.getElementById('export-png').addEventListener('click', () => {
    if (!state.network) { showError('Please load a knowledge graph first'); return; }
    try {
      const canvas = state.network.canvas.frame.canvas;
      const link = document.createElement('a');
      link.download = `kg_${state.currentKGName || 'graph'}_${new Date().toISOString().slice(0, 10)}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    } catch (error) {
      showError('PNG export failed: ' + error.message);
    }
  });

  // Export JSON (lines 4735-4738)
  document.getElementById('export-svg').addEventListener('click', () => {
    if (!state.graphData) { showError('Please load a knowledge graph first'); return; }
    exportGraphData();
  });

  // Node detail panel close button (lines 4457-4459)
  document.getElementById('node-panel-close').addEventListener('click', () => {
    document.getElementById('node-detail-panel').classList.remove('open');
  });
}
```

- [ ] **Step 3: Verify build**

```bash
cd frontend && npm run build
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/js/graph/filters.js frontend/src/js/graph/controls.js
git commit -m "feat: create graph filters and controls modules"
```

---

### Task 13: Create `chat/messages.js`

**Files:**

- Create: `frontend/src/js/chat/messages.js`

- [ ] **Step 1: Create `frontend/src/js/chat/messages.js`**

Extract these functions:

- `addToChat` (lines 3417-3494)
- Chat history restore (lines 4467-4479)
- Clear chat (lines 4556-4566)
- Export chat as markdown (lines 4755-4778)

```js
import { state } from '../state.js';
import { showError } from '../ui/notifications.js';

export function addToChat(message, type, id = null, skipHistory = false) {
  // ... extract from lines 3417-3494
  // Replace references to currentKGName with state.currentKGName where used in localStorage
}

export function initChat() {
  // Clear chat on page load
  const chatBox = document.getElementById('chat-box');
  if (chatBox) chatBox.innerHTML = '';

  // Setup error popup close button (lines 3578-3580)
  document.getElementById('close-error-popup').addEventListener('click', () => {
    document.getElementById('error-popup').classList.add('hidden');
  });

  // Restore chat history (lines 4467-4479)
  try {
    const history = JSON.parse(localStorage.getItem('kg-chat-history') || '[]');
    if (history.length > 0) {
      const emptyState = document.getElementById('chat-empty-state');
      if (emptyState) emptyState.style.display = 'none';
      history.forEach(entry => {
        addToChat(entry.message, entry.type, null, true);
      });
    }
  } catch (_e) { /* ignore */ }

  // Clear chat button (lines 4556-4566)
  document.getElementById('clear-chat-btn').addEventListener('click', () => {
    const chatBox = document.getElementById('chat-box');
    chatBox.innerHTML = '';
    const emptyState = document.getElementById('chat-empty-state');
    if (emptyState) {
      chatBox.appendChild(emptyState);
      emptyState.style.display = '';
    }
    localStorage.removeItem('kg-chat-history');
  });

  // Export chat as markdown (lines 4755-4778)
  document.getElementById('export-chat-btn').addEventListener('click', () => {
    const chatBox = document.getElementById('chat-box');
    const containers = chatBox.querySelectorAll('.message-container');
    if (!containers.length) { showError('No messages to export'); return; }
    let md = `# Chat Export — ${state.currentKGName || 'Knowledge Graph'}\n_Exported ${new Date().toLocaleString()}_\n\n`;
    containers.forEach(container => {
      const bubble = container.querySelector('.message-bubble');
      if (!bubble) return;
      const isUser = container.classList.contains('user');
      const isAI = container.classList.contains('ai');
      if (!isUser && !isAI) return;
      const clone = bubble.cloneNode(true);
      clone.querySelectorAll('.copy-msg-btn, .message-timestamp').forEach(el => el.remove());
      const text = (clone.innerText || clone.textContent || '').trim();
      if (!text) return;
      md += isUser ? `**You:** ${text}\n\n` : `**Assistant:** ${text}\n\n---\n\n`;
    });
    const blob = new Blob([md], { type: 'text/markdown' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `chat_${(state.currentKGName || 'export').replace(/\s+/g, '_')}_${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(a.href);
  });
}
```

- [ ] **Step 2: Verify build**

```bash
cd frontend && npm run build
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/js/chat/messages.js
git commit -m "feat: create chat messages module"
```

---

### Task 14: Create `chat/rag.js`

**Files:**

- Create: `frontend/src/js/chat/rag.js`

- [ ] **Step 1: Create `frontend/src/js/chat/rag.js`**

Extract:

- `sendQuestion` (window.sendQuestion, lines 4152-4425)
- `formatReasoningPath` (lines 4392-4397)
- `formatMarkdown` (lines 4400-4409)

```js
import { state } from '../state.js';
import { api } from '../api.js';
import { addToChat } from './messages.js';
import { initializeGraph } from '../graph/network.js';
import { applyHighlightStyles, updateHighlightBadge } from '../graph/legend.js';

function formatReasoningPath(text) {
  return text
    .replace(/\n/g, '<br>')
    .replace(/[→—>⇒]/g, '<span class="reason-arrow">→</span>')
    .replace(/\b(triggers|yields|reveals|leads to|results in)\b/gi, '<span class="reason-connector">$1</span>');
}

function formatMarkdown(text) {
  return text
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.*?)\*\*/g, '<strong class="md-strong">$1</strong>')
    .replace(/\*(.*?)\*/g, '<em class="md-em">$1</em>')
    .replace(/`(.*?)`/g, '<code class="md-code">$1</code>')
    .replace(/[•·]/g, '<br>• ')
    .replace(/^[\s]*[•·]\s*/gm, '<li style="margin:6px 0;margin-left:18px;">')
    .replace(/(\n|^)(\d+)\.\s+/gm, '$1$2. ');
}

export function initRAG() {
  // sendQuestion logic — extracted from lines 4152-4425
  window.sendQuestion = async function () {
    const input = document.getElementById('question');
    const sendButton = document.getElementById('send-btn');
    const question = input.value.trim();
    if (!question) return;

    const vendor = document.getElementById('rag-vendor').value;
    const model = document.getElementById('rag-model').value;

    addToChat(`You: ${question}`, 'user');
    input.value = '';

    const thinkingEl = addToChat(
      '<span class="btn-dots"><span></span><span></span><span></span></span>',
      'thinking', 'chat-thinking'
    );

    const originalButtonText = sendButton.textContent;
    try {
      sendButton.disabled = true;

      const payload = { question, provider_rag: vendor, model_rag: model };
      if (state.currentKGName) payload.kg_name = state.currentKGName;

      const chatAbort = new AbortController();
      const chatTimeout = setTimeout(() => chatAbort.abort(), 130000);
      let result;
      try {
        result = await api.sendChat(payload, chatAbort.signal);
      } finally {
        clearTimeout(chatTimeout);
      }

      // Update highlighted nodes from used_entities
      state.highlightedNodes.clear();
      const usedEntities = result.info?.entities?.used_entities || [];
      usedEntities.forEach(entity => {
        const readable = (entity.description || '').toLowerCase().trim();
        const idKey = (entity.id || '').toLowerCase().trim();
        if (readable) state.highlightedNodes.add(readable);
        if (idKey) state.highlightedNodes.add(idKey);
      });
      updateHighlightBadge(usedEntities.length);

      if (state.network) {
        applyHighlightStyles();
      } else if (state.graphData) {
        initializeGraph(state.graphData);
      }

      // ... rest of response formatting logic (lines 4228-4413)
      // Extract the full section parsing, formatted sections building,
      // source chip, sources panel, and strip [Source:] citations
      // exactly as in the original.
      // Use formatMarkdown() and formatReasoningPath() defined above.

      let formattedResponse = result.response || result.message || 'No response generated';

      let sections = { recommendation: '', reasoning: '', evidence: '', nextSteps: '' };
      const lines = formattedResponse.split('\n');
      let currentSection = '';
      let currentContent = [];

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line.toUpperCase().includes('RECOMMENDATION/SUMMARY') && line.includes('#')) {
          if (currentSection && currentContent.length > 0) { sections[currentSection] = currentContent.join('\n').trim(); currentContent = []; }
          currentSection = 'recommendation';
        } else if (line.toUpperCase().includes('REASONING PATH') && line.includes('#')) {
          if (currentSection && currentContent.length > 0) { sections[currentSection] = currentContent.join('\n').trim(); currentContent = []; }
          currentSection = 'reasoning';
        } else if (line.toUpperCase().includes('COMBINED EVIDENCE') && line.includes('#')) {
          if (currentSection && currentContent.length > 0) { sections[currentSection] = currentContent.join('\n').trim(); currentContent = []; }
          currentSection = 'evidence';
        } else if (line.toUpperCase().includes('NEXT STEPS') && line.includes('#')) {
          if (currentSection && currentContent.length > 0) { sections[currentSection] = currentContent.join('\n').trim(); currentContent = []; }
          currentSection = 'nextSteps';
        } else if (currentSection && line) {
          currentContent.push(line);
        }
      }
      if (currentSection && currentContent.length > 0) { sections[currentSection] = currentContent.join('\n').trim(); }

      let formattedSections = [];

      if (sections.recommendation.trim()) {
        formattedSections.push(`<div class="response-section recommendation-section"><button class="section-toggle expanded" onclick="toggleSection(this)">▾ Summary</button><div class="section-content expanded">${formatMarkdown(sections.recommendation)}</div></div>`);
      }
      if (sections.evidence.trim()) {
        formattedSections.push(`<div class="response-section evidence-section"><button class="section-toggle expanded" onclick="toggleSection(this)">▾ Evidence</button><div class="section-content expanded">${formatMarkdown(sections.evidence)}</div></div>`);
      }
      if (sections.nextSteps.trim()) {
        formattedSections.push(`<div class="response-section next-steps-section"><button class="section-toggle" onclick="toggleSection(this)">▸ Next steps</button><div class="section-content collapsed">${formatMarkdown(sections.nextSteps)}</div></div>`);
      }
      if (sections.reasoning.trim()) {
        formattedSections.push(`<div class="response-section reasoning-section"><button class="section-toggle" onclick="toggleSection(this)">▸ Reasoning path</button><div class="section-content collapsed">${formatReasoningPath(sections.reasoning)}</div></div>`);
      }
      if (formattedSections.length === 0) {
        formattedSections.push(`<div class="response-section full-response">${formatMarkdown(formattedResponse)}</div>`);
      }

      formattedResponse = formattedSections.join('<br>');

      const confidence = result.info?.confidence_score;
      const entityCount = (result.info?.entities?.used_entities || []).length;
      if (entityCount > 0) {
        const pct = confidence !== undefined ? `${Math.round(confidence * 100)}% confidence` : '';
        const src = `${entityCount} source${entityCount !== 1 ? 's' : ''}`;
        const chipText = [src, pct].filter(Boolean).join(' · ');
        formattedResponse = `<div class="source-chip">◈ ${chipText}</div>` + formattedResponse;
      }

      const reasoningEdges = result.info?.entities?.reasoning_edges || [];
      const sourceEntities = result.info?.entities?.used_entities || [];
      if (reasoningEdges.length > 0 || sourceEntities.length > 0) {
        let sourceLines = '';
        const seenEdges = new Set();
        reasoningEdges.forEach(edge => {
          const fromName = edge.from_name || edge.from || '?';
          const toName = edge.to_name || edge.to || '?';
          const rel = (edge.relationship || 'CONNECTED_TO').replace(/_/g, ' ');
          const key = `${fromName}|${rel}|${toName}`;
          if (!seenEdges.has(key)) {
            seenEdges.add(key);
            sourceLines += `<div class="src-edge"><span class="src-node">${fromName}</span><span class="src-rel"> ──${rel}──▶ </span><span class="src-node">${toName}</span></div>`;
          }
        });
        if (!sourceLines) {
          const names = [...new Set(sourceEntities.map(e => e.description || e.id).filter(Boolean))];
          sourceLines = names.map(n => `<span class="src-node">${n}</span>`).join(', ');
        }
        formattedResponse += `<div class="response-section sources-section"><button class="section-toggle" onclick="toggleSection(this)">▸ Sources</button><div class="section-content collapsed"><div class="src-edges-list">${sourceLines}</div></div></div>`;
      }

      formattedResponse = formattedResponse.replace(/【Source:[^】]*】/g, '').replace(/\[Source:[^\]]*\]/g, '');

      if (thinkingEl) thinkingEl.remove();
      addToChat(`${formattedResponse}`, 'ai');
    } catch (error) {
      if (thinkingEl) thinkingEl.remove();
      const msg = error.name === 'AbortError'
        ? 'Request timed out — the model took too long. Try a faster model or a shorter question.'
        : `Error: ${error.message}`;
      addToChat(msg, 'error');
    } finally {
      sendButton.innerHTML = originalButtonText;
      sendButton.disabled = false;
    }
  };

  // Enter key handler (lines 4428-4433)
  document.getElementById('question').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      window.sendQuestion();
    }
  });
}
```

- [ ] **Step 2: Verify build**

```bash
cd frontend && npm run build
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/js/chat/rag.js
git commit -m "feat: create chat RAG module"
```

---

### Task 15: Create `kg/models.js` and `kg/crud.js`

**Files:**

- Create: `frontend/src/js/kg/models.js`
- Create: `frontend/src/js/kg/crud.js`

- [ ] **Step 1: Create `frontend/src/js/kg/models.js`**

Extract `setupModelDropdowns` and `updateModelDropdown` (lines 3603-3672):

```js
import { api } from '../api.js';

export async function updateModelDropdown(vendorSelectId, modelSelectId) {
  const vendorSelect = document.getElementById(vendorSelectId);
  const modelSelect = document.getElementById(modelSelectId);
  const vendor = vendorSelect.value;

  modelSelect.innerHTML = '<option value="">Loading models...</option>';

  try {
    const data = await api.fetchModels(vendor);
    modelSelect.innerHTML = '';

    try {
      data.models.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        let displayName = model;
        if (vendor === 'openrouter') {
          const parts = model.split('/');
          if (parts.length > 1) displayName = parts[parts.length - 1];
          displayName = displayName.split(':')[0];
        }
        option.textContent = displayName;
        modelSelect.appendChild(option);
      });

      if (data.models.length > 0) {
        if (vendor === 'openrouter' && data.models.includes('openai/gpt-oss-120b:free')) {
          modelSelect.value = 'openai/gpt-oss-120b:free';
        } else {
          modelSelect.value = data.models[0];
        }
      } else {
        modelSelect.innerHTML = '<option value="">No models available</option>';
      }
    } catch (error) {
      console.error('Error processing models:', error);
      modelSelect.innerHTML = '<option value="">Error processing models</option>';
    }
  } catch (error) {
    console.error('Error fetching models:', error);
    modelSelect.innerHTML = '<option value="">Error loading models</option>';
  }
}

export function initModels() {
  updateModelDropdown('kg-provider', 'kg-model');
  updateModelDropdown('rag-vendor', 'rag-model');

  document.getElementById('kg-provider').addEventListener('change', () => {
    updateModelDropdown('kg-provider', 'kg-model');
  });

  document.getElementById('rag-vendor').addEventListener('change', () => {
    updateModelDropdown('rag-vendor', 'rag-model');
  });
}
```

- [ ] **Step 2: Create `frontend/src/js/kg/crud.js`**

Extract these functions:

- `storeKGId` (lines 3497-3500)
- `startProgressStream`, `stopProgressStream`, `appendLog` (lines 3522-3570)
- `loadKGList` (lines 3675-3713)
- KG name create dropdown handler (lines 3716-3730)
- `handleCreateKG` (lines 3907-4009)
- `handleConnectNeo4j` (lines 4012-4146)
- File upload handlers (lines 3847-3896)
- Ontology upload handlers (lines 3871-3886)
- `attachCreateKGHandler` (lines 3899-3905)
- Neo4j form open/close (lines 3852-3866)
- Neo4j default credentials prefill (lines 3591-3600)
- Clear KG button (lines 4485-4553)
- Progress panel close (lines 4462-4464)

```js
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

function storeKGId(kgId) {
  localStorage.setItem('currentKGId', kgId);
  console.log('Stored KG ID:', kgId);
}

let _progressSource = null;

function startProgressStream() {
  // ... extract from lines 3524-3560
}

function stopProgressStream() {
  if (_progressSource) { _progressSource.close(); _progressSource = null; }
}

function appendLog(logEl, text, cls) {
  // ... extract from lines 3564-3570
}

async function loadKGList() {
  // ... extract from lines 3675-3713
  // Replace direct fetch with api.fetchKGList()
}

async function handleCreateKG() {
  // ... extract from lines 3907-4009
  // Replace globals with state.* references
  // Replace direct fetch with api.createKG(formData)
  // Call updateKGBadge, updateChatKGName, storeKGId, initializeGraph, addToChat
}

async function handleConnectNeo4j() {
  // ... extract from lines 4012-4146
  // Replace globals with state.* references
  // Replace direct fetch with api.loadFromNeo4j(formData)
  // Call updateKGBadge, updateChatKGName, initializeGraph, addToChat, updateHighlightBadge
}

export function initKG() {
  // Pre-fill Neo4j defaults (lines 3591-3600)
  api.fetchDefaultCredentials()
    .then(data => {
      if (data.uri) document.getElementById('neo4j-uri').value = data.uri;
      if (data.user) document.getElementById('neo4j-user').value = data.user;
      if (data.uri) document.getElementById('save-neo4j-uri').value = data.uri;
      if (data.user) document.getElementById('save-neo4j-user').value = data.user;
    })
    .catch(() => {});

  // Load KG list on page load
  loadKGList();

  // KG name create dropdown handler (lines 3716-3730)
  document.getElementById('kg-name-create').addEventListener('change', function () {
    const input = document.getElementById('kg-name-new');
    if (this.value === '') {
      input.style.display = 'block';
      input.required = true;
    } else {
      input.style.display = 'none';
      input.required = false;
      input.value = '';
    }
  });

  // File upload triggers (lines 3847-3896)
  document.getElementById('load-file').addEventListener('click', () => {
    document.getElementById('file-upload').click();
  });

  document.getElementById('file-upload').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    document.getElementById('selected-file').textContent = `Selected: ${file.name}`;
    document.getElementById('file-selection').style.display = 'block';
  });

  // Ontology upload (lines 3871-3886)
  document.getElementById('select-ontology').addEventListener('click', () => {
    document.getElementById('ontology-upload').click();
  });

  document.getElementById('ontology-upload').addEventListener('change', (e) => {
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

  // Neo4j form open/close (lines 3852-3866)
  document.getElementById('load-neo4j').addEventListener('click', () => {
    loadKGList();
    document.getElementById('neo4j-form').style.display = 'block';
  });

  document.getElementById('close-neo4j-form').addEventListener('click', () => {
    document.getElementById('neo4j-form').style.display = 'none';
  });

  document.getElementById('close-save-neo4j-form').addEventListener('click', () => {
    document.getElementById('save-neo4j-form').style.display = 'none';
  });

  // Connect neo4j button
  document.getElementById('connect-neo4j').addEventListener('click', handleConnectNeo4j);

  // Create KG handler
  const createBtn = document.getElementById('create-kg-btn');
  createBtn.addEventListener('click', handleCreateKG);

  // Progress panel close (lines 4462-4464)
  document.getElementById('progress-panel-close').addEventListener('click', () => {
    document.getElementById('kg-progress-panel').style.display = 'none';
  });

  // Clear KG button (lines 4485-4553)
  document.getElementById('clear-kg-btn').addEventListener('click', async () => {
    if (!confirm('⚠️ WARNING: This will permanently delete ALL nodes and relationships from Neo4j database. This action cannot be undone. Are you sure you want to continue?')) return;

    const clearButton = document.getElementById('clear-kg-btn');
    const originalText = clearButton.textContent;

    try {
      clearButton.innerHTML = '<div class="spinner"></div> Clearing...';
      clearButton.disabled = true;

      const result = await api.clearKG();

      if (state.network) {
        try { state.network.destroy(); state.network = null; } catch (_e) { state.network = null; }
      }

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
      document.getElementById('overview-panel').style.display = 'none';
      await loadKGList();
      addToChat(`🧹 ${result.message}`, 'ai');
      showSuccess('Knowledge graph cleared successfully!');
    } catch (error) {
      console.error('Error clearing KG:', error);
      showError(`Failed to clear knowledge graph: ${error.message}`);
      addToChat(`Error clearing KG: ${error.message}`, 'error');
    } finally {
      clearButton.innerHTML = originalText;
      clearButton.disabled = false;
    }
  });
}
```

- [ ] **Step 3: Verify build**

```bash
cd frontend && npm run build
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/js/kg/
git commit -m "feat: create KG models and crud modules"
```

---

### Task 16: Wire up `main.js` entry point

**Files:**

- Modify: `frontend/src/main.js`

- [ ] **Step 1: Update `frontend/src/main.js`**

Replace the placeholder content with the full entry point:

```js
import './css/base.css';
import './css/animations.css';
import './css/layout.css';
import './css/graph.css';
import './css/chat.css';
import './css/components.css';

import { initTheme } from './js/ui/theme.js';
import { initSidebar } from './js/ui/sidebar.js';
import { initShortcuts } from './js/ui/shortcuts.js';
import { initHealth } from './js/ui/health.js';
import { initControls } from './js/graph/controls.js';
import { initFilters } from './js/graph/filters.js';
import { initChat } from './js/chat/messages.js';
import { initRAG } from './js/chat/rag.js';
import { initKG } from './js/kg/crud.js';
import { initModels } from './js/kg/models.js';

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initSidebar();
  initShortcuts();
  initModels();
  initKG();
  initControls();
  initFilters();
  initChat();
  initRAG();
  initHealth();
});
```

- [ ] **Step 2: Verify build**

```bash
cd frontend && npm run build
```

Expected: Build succeeds with zero errors. All imports resolve.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/main.js
git commit -m "feat: wire up main.js entry point with all module imports"
```

---

### Task 17: Update FastAPI to serve from `frontend/dist`

**Files:**

- Modify: `ontographrag/api/app.py` (lines 197, 468)

- [ ] **Step 1: Build the frontend**

```bash
cd frontend && npm run build
```

Verify `frontend/dist/index.html` exists.

- [ ] **Step 2: Update `app.py` StaticFiles mount**

In `ontographrag/api/app.py`, change line 197:

```python
# Before:
app.mount("/static", StaticFiles(directory="ontographrag/api/static"), name="static")

# After:
app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")
```

- [ ] **Step 3: Update `app.py` root route**

In `ontographrag/api/app.py`, change line 468:

```python
# Before:
@app.get("/")
async def root():
    return FileResponse("ontographrag/api/static/index.html")

# After:
@app.get("/")
async def root():
    return FileResponse("frontend/dist/index.html")
```

- [ ] **Step 4: Also serve Vite's built assets at root level**

Vite outputs JS/CSS assets to `frontend/dist/assets/`. Add a mount for this:

```python
app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")
```

Add this line right after the existing `/static` mount.

- [ ] **Step 5: Verify the static CSV template is still accessible**

Check if `ontographrag/api/static/` contains files other than `index.html` that need to stay:

```bash
ls ontographrag/api/static/
```

If there's a `medical_reports_template.csv` or similar, keep the old static mount as well or move those files to `frontend/public/static/`.

The route at line 1597 (`/static/medical_reports_template.csv`) expects the file at the old path. Either:

- Keep a second mount: `app.mount("/static/legacy", StaticFiles(directory="ontographrag/api/static"), name="legacy-static")` and update the CSV route, OR
- Move the CSV to `frontend/public/static/` so Vite copies it to `dist/static/`

Check what exists and choose the simplest approach.

- [ ] **Step 6: Commit**

```bash
git add ontographrag/api/app.py
git commit -m "feat: update FastAPI to serve frontend from frontend/dist"
```

---

### Task 18: Verify end-to-end and clean up

**Files:**

- Delete: `ontographrag/api/static/index.html` (after verification)

- [ ] **Step 1: Build frontend**

```bash
cd frontend && npm run build
```

Expected: Zero errors.

- [ ] **Step 2: Start FastAPI server**

```bash
.venv/bin/python3 -m uvicorn ontographrag.api.app:app --host 0.0.0.0 --port 8000
```

- [ ] **Step 3: Smoke test the production build**

Open `http://localhost:8000` in a browser and verify:

1. Page loads without console errors
2. Theme toggle switches between dark/light
3. Sidebar collapses/expands
4. Health dot shows green/yellow/red
5. Model dropdowns populate when vendor is changed
6. Neo4j form opens and closes
7. KG can be loaded from Neo4j (if available)
8. Graph renders with nodes and edges
9. Search highlights matching nodes
10. Zoom in/out/reset work
11. Filters show/hide node types
12. Chat input accepts questions
13. Keyboard shortcuts work (Esc, Cmd+K)
14. Export PNG/JSON buttons work

- [ ] **Step 4: Test Vite dev server**

In a separate terminal:

```bash
cd frontend && npm run dev
```

Open `http://localhost:5173` and verify the same smoke tests pass with hot module replacement working.

- [ ] **Step 5: Delete old index.html**

Once verified, delete the old monolithic file:

```bash
rm ontographrag/api/static/index.html
```

If `ontographrag/api/static/` is now empty, the `StaticFiles` mount for `/static` should either be removed or pointed to `frontend/dist`. If other files remain (CSV template), keep the mount.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "chore: remove old monolithic index.html after frontend extraction"
```

