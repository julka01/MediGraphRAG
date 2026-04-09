import { useCallback, useRef } from 'react';

interface UseSnapToCloseOptions {
  edge: 'left' | 'right' | 'bottom';
  minSize: number;
  onClose: () => void;
  onOpen: () => void;
  onResize: (size: number) => void;
  snapThreshold?: number;
}

export function useSnapToClose({ edge, minSize, onClose, onOpen, onResize, snapThreshold = 40 }: UseSnapToCloseOptions) {
  const dragging = useRef(false);
  const snapped = useRef(false);

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    dragging.current = true;
    snapped.current = false;
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

    if (nearEdge && !snapped.current) {
      snapped.current = true;
      onClose();
      return;
    }

    if (!nearEdge && snapped.current) {
      snapped.current = false;
      onOpen();
    }

    if (!snapped.current) {
      const maxSize = edge === 'bottom' ? windowHeight / 2 : windowWidth / 2;
      const clamped = Math.max(minSize, Math.min(maxSize, newSize));
      onResize(clamped);
    }
  }, [edge, minSize, onClose, onOpen, onResize, snapThreshold]);

  const onPointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    dragging.current = false;
    snapped.current = false;
    (e.target as HTMLElement).releasePointerCapture(e.pointerId);
    document.body.classList.remove('select-none');
  }, []);

  return { onPointerDown, onPointerMove, onPointerUp, dragging };
}
