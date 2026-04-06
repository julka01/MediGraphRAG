import { state } from '../state.js';

export function initSidebar() {
  const kgView = document.getElementById('kg-view');
  const chatView = document.getElementById('chat-view');

  // Default layout: both panels visible, KG at 62% / chat at 38%
  kgView.style.flex = '1.6';
  chatView.style.flex = '1';

  // KG expand / collapse toggle
  let kgExpanded = false;
  const expandBtn = document.getElementById('kg-expand-btn');
  const expandIconCollapse = `<svg viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" width="14" height="14"><path d="M6 1v5H1M15 6h-5V1M10 15v-5h5M1 10h5v5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  const expandIconExpand  = `<svg viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" width="14" height="14"><path d="M1 6V1h5M10 1h5v5M15 10v5h-5M6 15H1v-5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

  expandBtn.addEventListener('click', function () {
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
    // Give the layout time to settle, then redraw
    if (state.network) setTimeout(() => state.network.redraw(), 150);
  });

  // Sidebar collapse toggle
  document.getElementById('sidebar-toggle').addEventListener('click', function () {
    const panel = document.querySelector('.left-panel');
    panel.classList.toggle('collapsed');
    const collapsed = panel.classList.contains('collapsed');
    this.textContent = collapsed ? '›' : '‹';
    this.title = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
    // Redraw network after layout shift
    if (state.network) setTimeout(() => state.network.redraw(), 300);
  });
}
