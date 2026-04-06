import { performSearch } from '../graph/filters.js';

export function initShortcuts() {
  // Esc: clear node search and close node detail panel
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
      const searchInput = document.getElementById('node-search');
      if (searchInput && searchInput.value) {
        searchInput.value = '';
        performSearch('');
      }
      // Also close node detail panel
      const ndPanel = document.getElementById('node-detail-panel');
      if (ndPanel) ndPanel.classList.remove('open');
    }
  });

  // Cmd+K / Ctrl+K focuses the chat input
  document.addEventListener('keydown', function (e) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      const q = document.getElementById('question');
      if (q) { q.focus(); q.select(); }
    }
  });

  // Suggestion chips: fill textarea and focus
  window.useSuggestion = function (btn) {
    const q = document.getElementById('question');
    if (q) { q.value = btn.textContent.trim(); q.focus(); }
  };

  // Toggle response sections expanded/collapsed
  window.toggleSection = function (button) {
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

  // Toggle reasoning steps visibility
  window.toggleReasoningSteps = function () {
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
