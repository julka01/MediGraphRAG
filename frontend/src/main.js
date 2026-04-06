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
