import { useEffect, useState } from 'react';

/**
 * Measures the natural (unshrunk) width of [data-chat-controls] and locks it
 * as the sidebar's minimum width so controls never squeeze into each other.
 *
 * Uses ResizeObserver on the controls element instead of MutationObserver on
 * the entire subtree — fires only when size actually changes.
 */
export function useDynamicMinWidth(containerRef: React.RefObject<HTMLElement | null>) {
  const [minWidth, setMinWidth] = useState(280);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const controlsRow = container.querySelector('[data-chat-controls]') as HTMLElement | null;
    if (!controlsRow) return;

    function measure() {
      if (!controlsRow) return;
      const prev = controlsRow.style.width;
      controlsRow.style.width = 'max-content';
      const natural = controlsRow.scrollWidth + 24;
      controlsRow.style.width = prev;
      setMinWidth(Math.max(240, natural));
    }

    const observer = new ResizeObserver(measure);
    observer.observe(controlsRow);
    measure();

    return () => observer.disconnect();
  }, [containerRef]);

  return minWidth;
}
