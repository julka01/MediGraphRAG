import { useSnapToClose } from '../../hooks/useSnapToClose';

interface BottomResizeHandleProps {
  onResize: (height: number) => void;
  onClose: () => void;
  minHeight: number;
}

export function BottomResizeHandle({ onResize, onClose, minHeight }: BottomResizeHandleProps) {
  const { handleRef, onPointerDown, onPointerMove, onPointerUp, dragging } = useSnapToClose({
    edge: 'bottom',
    minSize: minHeight,
    onClose,
    onResize,
  });

  return (
    <div
      ref={handleRef}
      role="separator"
      aria-orientation="horizontal"
      aria-label="Resize bottom panel"
      className={`h-1 shrink-0 cursor-row-resize transition-colors ${
        dragging.current ? 'bg-primary/50' : 'bg-base-300 hover:bg-primary/50'
      }`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    />
  );
}
