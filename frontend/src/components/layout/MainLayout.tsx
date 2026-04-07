import { ArrowLeftStartOnRectangleIcon, ArrowRightStartOnRectangleIcon } from '@heroicons/react/20/solid';
import clsx from 'clsx';
import { type ReactNode, useCallback, useRef, useState } from 'react';
import { useApp } from '../../context/AppContext';
import type { Layout } from '../../types/app';
import { safeGet, safeSet } from '../../utils/storage';
import { ResizeHandle } from './ResizeHandle';

const STORAGE_KEY = 'kg-split-width';
const DEFAULT_WIDTH = 67;

function readStoredWidth(): number {
  const stored = safeGet(STORAGE_KEY);
  if (!stored) return DEFAULT_WIDTH;
  const parsed = Number(stored);
  return Number.isFinite(parsed) ? parsed : DEFAULT_WIDTH;
}

interface MainLayoutProps {
  layout: Layout;
  graphPanel: ReactNode;
  chatPanel: ReactNode;
}

export function MainLayout({ layout, graphPanel, chatPanel }: MainLayoutProps) {
  const { state, dispatch } = useApp();
  const showGraph = layout !== 'chat-only';
  const showChat = layout !== 'graph-only';
  const isSplit = showGraph && showChat;

  const [graphWidth, setGraphWidth] = useState(readStoredWidth);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleResize = useCallback((pct: number) => {
    setGraphWidth(pct);
    safeSet(STORAGE_KEY, String(Math.round(pct)));
  }, []);

  const handleReset = useCallback(() => {
    setGraphWidth(DEFAULT_WIDTH);
    safeSet(STORAGE_KEY, String(DEFAULT_WIDTH));
  }, []);

  const graphStyle = isSplit ? { width: `${graphWidth}%` } : undefined;
  const graphClass = clsx(
    'border-b md:border-b-0 md:border-r border-base-300 overflow-hidden',
    isSplit ? 'shrink-0 max-md:flex-1' : 'flex-1',
  );

  return (
    <div ref={containerRef} className="flex flex-col md:flex-row flex-1 min-w-0">
      {showGraph && (
        <div className={clsx('@container', graphClass)} style={graphStyle}>
          {graphPanel}
        </div>
      )}
      {isSplit && (
        <ResizeHandle
          onResize={handleResize}
          onDoubleClick={handleReset}
          containerRef={containerRef}
          valuenow={graphWidth}
        />
      )}
      <div className="hidden md:flex items-center justify-center shrink-0">
        <button
          type="button"
          className="btn btn-ghost btn-xs btn-square"
          onClick={() => dispatch({ type: 'TOGGLE_KG_EXPANDED' })}
          title={state.kgExpanded ? 'Open chat' : 'Close chat'}
          aria-label={state.kgExpanded ? 'Open chat' : 'Close chat'}
        >
          {state.kgExpanded ? (
            <ArrowLeftStartOnRectangleIcon className="size-4" aria-hidden="true" />
          ) : (
            <ArrowRightStartOnRectangleIcon className="size-4" aria-hidden="true" />
          )}
        </button>
      </div>
      <div className="hidden md:block w-px shrink-0 bg-base-300" />
      {showChat && <div className="@container overflow-hidden flex-1">{chatPanel}</div>}
    </div>
  );
}
