import { type ReactNode, useCallback, useRef, useState } from 'react';
import type { Layout } from '../../types/app';
import { ResizeHandle } from './ResizeHandle';

const STORAGE_KEY = 'kg-split-width';
const DEFAULT_WIDTH = 67;

function readStoredWidth(): number {
  const stored = localStorage.getItem(STORAGE_KEY);
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
  const showGraph = layout !== 'chat-only';
  const showChat = layout !== 'graph-only';
  const isSplit = showGraph && showChat;

  const [graphWidth, setGraphWidth] = useState(readStoredWidth);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleResize = useCallback((pct: number) => {
    setGraphWidth(pct);
    localStorage.setItem(STORAGE_KEY, String(Math.round(pct)));
  }, []);

  const handleReset = useCallback(() => {
    setGraphWidth(DEFAULT_WIDTH);
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  const graphStyle = isSplit ? { width: `${graphWidth}%` } : undefined;
  const graphClass = isSplit
    ? 'border-b md:border-b-0 md:border-r border-base-300 overflow-hidden shrink-0 max-md:flex-1'
    : 'border-b md:border-b-0 md:border-r border-base-300 overflow-hidden flex-1';

  return (
    <div ref={containerRef} className="flex flex-col md:flex-row flex-1 min-w-0">
      {showGraph && (
        <div className={graphClass} style={graphStyle}>
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
      {showChat && <div className="overflow-hidden flex-1">{chatPanel}</div>}
    </div>
  );
}
