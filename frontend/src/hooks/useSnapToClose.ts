import { useCallback, useRef } from 'react';

interface UseSnapToCloseOptions {
  edge: 'left' | 'right' | 'bottom';
  minSize: number;
  onClose: () => void;
  onResize: (size: number) => void;
  snapThreshold?: number; // px from window edge to trigger snap
}

export function useSnapToClose({ edge, minSize, onClose, onResize, snapThreshold = 40 }: UseSnapToCloseOptions) {
  const dragging = useRef(false);
  const handleRef = useRef<HTMLDivElement>(null);

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    dragging.current = true;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    document.body.classList.add('select-none');
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragging.current) return;

    const { clientX, clientY } = e;
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;

    let nearEdge = false;
    let newSize: number;

    if (edge === 'left') {
      newSize = clientX;
      nearEdge = clientX <= snapThreshold;
    } else if (edge === 'right') {
      newSize = windowWidth - clientX;
      nearEdge = clientX >= windowWidth - snapThreshold;
    } else {
      newSize = windowHeight - clientY;
      nearEdge = clientY >= windowHeight - snapThreshold;
    }

    if (nearEdge) {
      dragging.current = false;
      document.body.classList.remove('select-none');
      onClose();
      return;
    }

    const maxSize = edge === 'bottom' ? windowHeight / 2 : windowWidth / 2;
    const clamped = Math.max(minSize, Math.min(maxSize, newSize));
    onResize(clamped);
  }, [edge, minSize, onClose, onResize, snapThreshold]);

  const onPointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    dragging.current = false;
    (e.target as HTMLElement).releasePointerCapture(e.pointerId);
    document.body.classList.remove('select-none');
  }, []);

  return { handleRef, onPointerDown, onPointerMove, onPointerUp, dragging };
}
