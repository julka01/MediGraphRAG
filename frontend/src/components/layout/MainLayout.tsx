import { type ReactNode, useRef } from 'react';
import { useApp } from '../../context/AppContext';
import { useDynamicMinWidth } from '../../hooks/useDynamicMinWidth';
import { useSnapToClose } from '../../hooks/useSnapToClose';

interface MainLayoutProps {
  graphPanel: ReactNode;
  chatPanel: ReactNode;
  bottomBar: ReactNode;
  topBar: ReactNode;
}

export function MainLayout({ graphPanel, chatPanel, bottomBar, topBar }: MainLayoutProps) {
  const { state, dispatch } = useApp();
  const { rightCollapsed, bottomCollapsed, topCollapsed, rightWidth } = state.panels;
  const rightSidebarRef = useRef<HTMLDivElement>(null);
  const dynamicMinWidth = useDynamicMinWidth(rightSidebarRef);

  const rightSnap = useSnapToClose({
    edge: 'right',
    minSize: dynamicMinWidth,
    onClose: () => dispatch({ type: 'CLOSE_PANEL', payload: 'right' }),
    onOpen: () => dispatch({ type: 'OPEN_PANEL', payload: 'right' }),
    onResize: (w) => dispatch({ type: 'SET_RIGHT_WIDTH', payload: w }),
  });

  const bottomSnap = useSnapToClose({
    edge: 'bottom',
    minSize: 80,
    snapThreshold: 120,
    onClose: () => dispatch({ type: 'CLOSE_PANEL', payload: 'bottom' }),
    onOpen: () => dispatch({ type: 'OPEN_PANEL', payload: 'bottom' }),
    onResize: (h) => dispatch({ type: 'SET_BOTTOM_HEIGHT', payload: h }),
  });

  const topSnap = useSnapToClose({
    edge: 'top',
    minSize: 48,
    onClose: () => dispatch({ type: 'CLOSE_PANEL', payload: 'top' }),
    onOpen: () => dispatch({ type: 'OPEN_PANEL', payload: 'top' }),
    onResize: () => {}, // fixed height, no resize — only close gesture
  });

  return (
    <div className="flex flex-1 min-w-0 min-h-0">
      {/* Graph + Top + Bottom stack */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        {/* Top panel + snap handle */}
        {!topCollapsed && (
          <>
            {topBar}
            <div
              role="separator"
              aria-orientation="horizontal"
              aria-label="Drag to close top panel"
              className="app-divider-h h-1 shrink-0 cursor-row-resize transition-colors"
              onPointerDown={topSnap.onPointerDown}
            />
          </>
        )}

        {/* Graph view */}
        <div className="flex-1 min-h-0">{graphPanel}</div>

        {/* Bottom resize handle + bottom bar — always mounted to preserve filter state */}
        {!bottomCollapsed && (
          <div
            role="separator"
            aria-orientation="horizontal"
            aria-label="Resize bottom panel"
            className="app-divider-h h-1 shrink-0 cursor-row-resize transition-colors"
            onPointerDown={bottomSnap.onPointerDown}
          />
        )}
        <div className={bottomCollapsed ? 'hidden' : ''}>{bottomBar}</div>
      </div>

      {/* Right sidebar resize handle + chat panel — always mounted to preserve chat state */}
      {!rightCollapsed && (
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize right panel"
          className="app-divider-v hidden md:block w-1 shrink-0 cursor-col-resize transition-colors"
          onPointerDown={rightSnap.onPointerDown}
        />
      )}
      <div
        ref={rightSidebarRef}
        className={rightCollapsed ? 'hidden' : 'overflow-hidden'}
        style={rightCollapsed ? undefined : { width: Math.max(rightWidth, dynamicMinWidth), minWidth: dynamicMinWidth }}
      >
        {chatPanel}
      </div>
    </div>
  );
}
