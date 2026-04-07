import { useCallback, useRef, useState } from 'react';

const MIN_PCT = 20;
const MAX_PCT = 80;

interface ResizeHandleProps {
  onResize: (percentage: number) => void;
  onDoubleClick: () => void;
  containerRef: React.RefObject<HTMLDivElement | null>;
}

export function ResizeHandle({ onResize, onDoubleClick, containerRef }: ResizeHandleProps) {
  const [dragging, setDragging] = useState(false);
  const handleRef = useRef<HTMLDivElement>(null);

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.preventDefault();
      handleRef.current?.setPointerCapture(e.pointerId);
      setDragging(true);
      containerRef.current?.classList.add('select-none');
    },
    [containerRef],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!dragging) return;
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const pct = ((e.clientX - rect.left) / rect.width) * 100;
      onResize(Math.min(MAX_PCT, Math.max(MIN_PCT, pct)));
    },
    [dragging, containerRef, onResize],
  );

  const onPointerUp = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      handleRef.current?.releasePointerCapture(e.pointerId);
      setDragging(false);
      containerRef.current?.classList.remove('select-none');
    },
    [containerRef],
  );

  return (
    <div
      ref={handleRef}
      role="separator"
      aria-orientation="vertical"
      className={`hidden md:block w-1 shrink-0 cursor-col-resize transition-colors ${
        dragging ? 'bg-primary/50' : 'bg-base-300 hover:bg-primary/50'
      }`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onDoubleClick={onDoubleClick}
    />
  );
}
