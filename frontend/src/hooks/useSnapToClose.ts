import { useCallback, useEffect, useRef } from 'react';

interface UseSnapToCloseOptions {
  edge: 'left' | 'right' | 'bottom' | 'top';
  minSize: number;
  onClose: () => void;
  onOpen: () => void;
  onResize: (size: number) => void;
  snapThreshold?: number;
}

/**
 * Drag-to-resize with snap-to-close/open behavior.
 *
 * Move and up listeners are attached to `document` so they survive the
 * resize-handle element being unmounted mid-drag (which happens when the
 * panel snaps closed and React removes it from the tree).
 */
export function useSnapToClose({ edge, minSize, onClose, onOpen, onResize, snapThreshold = 40 }: UseSnapToCloseOptions) {
  const dragging = useRef(false);
  const snapped = useRef(false);

  // Keep latest callbacks in refs so the document listeners always call
  // the current version without needing to re-attach on every render.
  const cbRef = useRef({ onClose, onOpen, onResize });
  cbRef.current = { onClose, onOpen, onResize };

  const configRef = useRef({ edge, minSize, snapThreshold });
  configRef.current = { edge, minSize, snapThreshold };

  const handlePointerMove = useCallback((e: PointerEvent) => {
    if (!dragging.current) return;

    const { clientX, clientY } = e;
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    const { edge: ed, minSize: min, snapThreshold: thresh } = configRef.current;

    let nearEdge = false;
    let newSize: number;

    if (ed === 'left') {
      newSize = clientX;
      nearEdge = clientX <= thresh;
    } else if (ed === 'right') {
      newSize = windowWidth - clientX;
      nearEdge = clientX >= windowWidth - thresh;
    } else if (ed === 'top') {
      newSize = clientY;
      nearEdge = clientY <= thresh;
    } else {
      newSize = windowHeight - clientY;
      nearEdge = clientY >= windowHeight - thresh;
    }

    if (nearEdge && !snapped.current) {
      snapped.current = true;
      cbRef.current.onClose();
      return;
    }

    if (!nearEdge && snapped.current) {
      snapped.current = false;
      cbRef.current.onOpen();
    }

    if (!snapped.current) {
      const maxSize = (ed === 'bottom' || ed === 'top') ? windowHeight / 2 : windowWidth / 2;
      const clamped = Math.max(min, Math.min(maxSize, newSize));
      cbRef.current.onResize(clamped);
    }
  }, []);

  const handlePointerUp = useCallback(() => {
    if (!dragging.current) return;
    dragging.current = false;
    snapped.current = false;
    document.removeEventListener('pointermove', handlePointerMove);
    document.removeEventListener('pointerup', handlePointerUp);
    document.body.classList.remove('select-none');
  }, [handlePointerMove]);

  // Clean up document listeners if the hook unmounts mid-drag.
  useEffect(() => {
    return () => {
      if (dragging.current) {
        document.removeEventListener('pointermove', handlePointerMove);
        document.removeEventListener('pointerup', handlePointerUp);
        document.body.classList.remove('select-none');
        dragging.current = false;
      }
    };
  }, [handlePointerMove, handlePointerUp]);

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.preventDefault();
      dragging.current = true;
      snapped.current = false;
      document.body.classList.add('select-none');
      document.addEventListener('pointermove', handlePointerMove);
      document.addEventListener('pointerup', handlePointerUp);
    },
    [handlePointerMove, handlePointerUp],
  );

  return { onPointerDown, dragging };
}
