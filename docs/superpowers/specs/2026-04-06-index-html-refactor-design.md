# Design: Refactor index.html into Separate HTML, JS, CSS with Vite

**Date:** 2026-04-06
**Status:** Approved

## Overview

Refactor the monolithic `ontographrag/api/static/index.html` (4,809 lines) into a Vite-based frontend project with ES modules and separated CSS files. This is a pure structural refactor with no behavioral changes.

## Current State

- Single file: `ontographrag/api/static/index.html`
- ~1,900 lines inline CSS (`<style>` block, lines 12-1910)
- ~340 lines HTML markup (lines 1912-2246)
- ~2,560 lines inline JS (`<script>` block, lines 2248-4807)
- Served by FastAPI via `StaticFiles` mount and `FileResponse`
- External CDN deps: vis-network 9.1.0, Prism.js 1.29.0, Font Awesome 6.4.0

## Decisions

- **ES modules** with `import`/`export` (not plain script tags)
- **Vite** as build tool (dev server with proxy, production bundling)
- **Separate `frontend/` directory** at project root with its own `package.json`
- **Feature-based module split** for JS (not coarse/barrel)
- **Concern-based split** for CSS (6 files)
- FastAPI serves built output from `frontend/dist/`

## Project Structure

```
frontend/
├── package.json
├── vite.config.js
├── index.html
├── src/
│   ├── main.js
│   ├── css/
│   │   ├── base.css
│   │   ├── layout.css
│   │   ├── graph.css
│   │   ├── chat.css
│   │   ├── components.css
│   │   └── animations.css
│   └── js/
│       ├── state.js
│       ├── api.js
│       ├── graph/
│       │   ├── network.js
│       │   ├── controls.js
│       │   ├── filters.js
│       │   └── legend.js
│       ├── chat/
│       │   ├── messages.js
│       │   └── rag.js
│       ├── kg/
│       │   ├── crud.js
│       │   └── models.js
│       └── ui/
│           ├── theme.js
│           ├── notifications.js
│           ├── sidebar.js
│           ├── shortcuts.js
│           └── health.js
```

## Shared State Module

`state.js` exports a single mutable object that all modules import by reference:

```js
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
```

All modules do `import { state } from '../state.js'` and mutate `state.*` directly. No event bus or store library needed.

## API Module

`api.js` centralizes all `fetch` calls behind named async functions:

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

## Module Dependency Graph

Arrows mean "imports from":

```
main.js
 ├── state.js
 ├── api.js
 ├── ui/theme.js         → state
 ├── ui/notifications.js  (leaf - no project deps)
 ├── ui/sidebar.js        → state
 ├── ui/shortcuts.js       (leaf - DOM only)
 ├── graph/legend.js      → state
 ├── graph/filters.js     → state, graph/network
 ├── graph/network.js     → state, graph/legend, ui/theme, ui/notifications
 ├── graph/controls.js    → state, graph/network, ui/notifications
 ├── chat/messages.js     → state
 ├── chat/rag.js          → state, api, chat/messages, graph/network
 ├── kg/models.js         → api
 └── kg/crud.js           → state, api, graph/network, chat/messages, kg/models, ui/notifications
```

**Rules to prevent circular imports:**
- `state.js`, `api.js`, and `ui/notifications.js` import nothing from the project
- `graph/network.js` is the heaviest dependency; no module it imports may import it back
- `main.js` is the only module that wires `addEventListener` calls; other modules export functions but don't self-execute

## main.js Entry Point

```js
import './css/base.css';
import './css/layout.css';
import './css/graph.css';
import './css/chat.css';
import './css/components.css';
import './css/animations.css';

import { initTheme } from './js/ui/theme.js';
import { initSidebar } from './js/ui/sidebar.js';
import { initShortcuts } from './js/ui/shortcuts.js';
import { initGraph } from './js/graph/network.js';
import { initControls } from './js/graph/controls.js';
import { initFilters } from './js/graph/filters.js';
import { initChat } from './js/chat/messages.js';
import { initRAG } from './js/chat/rag.js';
import { initKG } from './js/kg/crud.js';
import { initModels } from './js/kg/models.js';
import { initHealth } from './js/ui/health.js';

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

## Vite Configuration

```js
// frontend/vite.config.js
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
    },
  },
});
```

**Dev workflow:**
- `cd frontend && npm run dev` — Vite dev server on port 5173, proxies API to FastAPI on 8000
- `cd frontend && npm run build` — outputs to `frontend/dist/`

## CSS Split

| File | Content | ~Lines |
|------|---------|--------|
| `base.css` | CSS custom properties, `*` reset, body font, `.hidden`, theme variable blocks | ~120 |
| `layout.css` | Body flex, `.left-panel`, `.right-panel`, `.kg-view`, `.chat-view`, sidebar collapse, KG expand | ~200 |
| `graph.css` | `#graph-container`, controls, legend, overview, filter panel, node detail panel, search, badges | ~550 |
| `chat.css` | Chat panel, messages, bubbles, input, suggestions, response sections, source chips | ~600 |
| `components.css` | Buttons, forms (neo4j), inputs/selects, progress panel, radio buttons, stats | ~350 |
| `animations.css` | Spinner, dot pulse, transitions | ~30 |

## JS Module Assignments

| Module | Functions | ~Lines |
|--------|-----------|--------|
| `state.js` | Shared mutable state object | ~20 |
| `api.js` | `request`, `api.*` fetch helpers | ~40 |
| `graph/network.js` | `initializeGraph`, node click handler, merge mode, vis-network creation | ~500 |
| `graph/controls.js` | `handleZoomIn/Out/Reset`, `handlePhysicsToggle`, `handleNodeSizeChange`, `createAdvancedButton`, `exportGraphData`, `findShortestPath`, export PNG/JSON | ~200 |
| `graph/filters.js` | `updateFilterPanel`, `applyFilters`, `performSearch` | ~150 |
| `graph/legend.js` | `generateNodeTypeColors`, `generateRelationshipTypeColors`, `updateLegend`, `updateOverview`, `updateMiniLegend`, `updateHighlightBadge`, `updateKGBadge`, `updateChatKGName`, `confidenceEdgeColor`, `applyHighlightStyles` | ~300 |
| `chat/messages.js` | `addToChat`, chat history persist/restore, clear chat, export chat as markdown | ~150 |
| `chat/rag.js` | `sendQuestion`, response section parsing, `formatMarkdown`, `formatReasoningPath`, entity/highlight handling | ~300 |
| `kg/crud.js` | `handleCreateKG`, `handleConnectNeo4j`, `loadKGList`, clear KG handler, file/ontology upload, KG name dropdown, `storeKGId`, save to neo4j, progress streaming (`startProgressStream`, `stopProgressStream`, `appendLog`) | ~450 |
| `kg/models.js` | `setupModelDropdowns`, `updateModelDropdown` | ~80 |
| `ui/theme.js` | Theme toggle, `getGraphTheme` | ~50 |
| `ui/notifications.js` | `showError`, `showSuccess` | ~20 |
| `ui/sidebar.js` | Sidebar collapse/expand, KG expand/collapse | ~50 |
| `ui/shortcuts.js` | Keyboard shortcuts (Esc, Cmd+K), `toggleSection`, `toggleReasoningSteps`, `useSuggestion` | ~50 |
| `ui/health.js` | `/doctor` health check on load, health dot click-to-recheck | ~40 |

## FastAPI Changes

In `ontographrag/api/app.py`:
- Change `StaticFiles` mount: `"ontographrag/api/static"` → `"frontend/dist"`
- Change `FileResponse`: `"ontographrag/api/static/index.html"` → `"frontend/dist/index.html"`
- Delete `ontographrag/api/static/index.html` after migration

## Migration Strategy

Single atomic migration (not incremental). The file is self-contained with no other consumers.

**Steps:**
1. Create `frontend/` scaffolding (package.json, vite.config.js, index.html)
2. Extract CSS into 6 files
3. Extract JS into modules (state -> api -> leaves -> dependents -> main.js)
4. Strip all `<style>` and `<script>` from index.html, add module script tag
5. Update `app.py` to serve from `frontend/dist`
6. Delete old `ontographrag/api/static/index.html`

## Testing Plan

- `npm run build` must succeed with zero errors
- Manual smoke test:
  - Theme toggle works (dark/light)
  - Sidebar collapse/expand
  - Model dropdowns populate on vendor change
  - Neo4j form opens/closes
  - Graph loads and renders from Neo4j
  - Search, filter, zoom, reset all work
  - Chat sends questions, receives responses
  - KG creation flow works
  - Keyboard shortcuts (Esc, Cmd+K)
  - Export PNG/JSON
- No behavioral changes expected

## Risks

- **EventSource for SSE** (`/kg_progress_stream`): Vite's dev proxy handles SSE natively, no special config needed
- **vis-network global `vis`**: Stays as CDN `<script>` tag (loaded before module), so `vis.Network` and `vis.DataSet` remain available as globals
