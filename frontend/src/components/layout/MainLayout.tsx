import { type ReactNode, useRef } from 'react';
import { BottomResizeHandle } from './BottomResizeHandle';
import { useApp } from '../../context/AppContext';
import { useDynamicMinWidth } from '../../hooks/useDynamicMinWidth';
import { useSnapToClose } from '../../hooks/useSnapToClose';

interface MainLayoutProps {
  graphPanel: ReactNode;
  chatPanel: ReactNode;
  bottomBar: ReactNode;
}

export function MainLayout({ graphPanel, chatPanel, bottomBar }: MainLayoutProps) {
  const { state, dispatch } = useApp();
  const { rightCollapsed, bottomCollapsed, rightWidth } = state.panels;
  const rightSidebarRef = useRef<HTMLDivElement>(null);
  const dynamicMinWidth = useDynamicMinWidth(rightSidebarRef);

  const rightSnap = useSnapToClose({
    edge: 'right',
    minSize: dynamicMinWidth,
    onClose: () => dispatch({ type: 'CLOSE_PANEL', payload: 'right' }),
    onOpen: () => dispatch({ type: 'OPEN_PANEL', payload: 'right' }),
    onResize: (w) => dispatch({ type: 'SET_RIGHT_WIDTH', payload: w }),
  });

  function handleBottomResize(height: number) {
    dispatch({ type: 'SET_BOTTOM_HEIGHT', payload: height });
  }

  function handleBottomClose() {
    dispatch({ type: 'CLOSE_PANEL', payload: 'bottom' });
  }

  return (
    <div className="flex flex-1 min-w-0 min-h-0">
      {/* Graph + Bottom stack */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Graph view */}
        <div className="flex-1 min-h-0">
          {graphPanel}
        </div>

        {/* Bottom resize handle + bottom bar */}
        {!bottomCollapsed && (
          <>
            <BottomResizeHandle
              onResize={handleBottomResize}
              onClose={handleBottomClose}
              onOpen={() => dispatch({ type: 'OPEN_PANEL', payload: 'bottom' })}
              minHeight={80}
            />
            {bottomBar}
          </>
        )}
      </div>

      {/* Right sidebar resize handle + chat panel */}
      {!rightCollapsed && (
        <>
          <div
            role="separator"
            className="hidden md:block w-1 shrink-0 cursor-col-resize transition-colors bg-base-300 hover:bg-primary/50"
            onPointerDown={rightSnap.onPointerDown}
            onPointerMove={rightSnap.onPointerMove}
            onPointerUp={rightSnap.onPointerUp}
          />
          <div
            ref={rightSidebarRef}
            className="shrink-0 overflow-hidden"
            style={{ width: Math.max(rightWidth, dynamicMinWidth) }}
          >
            {chatPanel}
          </div>
        </>
      )}
    </div>
  );
}
