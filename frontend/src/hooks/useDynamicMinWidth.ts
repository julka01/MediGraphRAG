import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Measures the natural (unshrunk) width of [data-chat-controls] and locks it
 * as the sidebar's minimum width so controls never squeeze into each other.
 */
export function useDynamicMinWidth(containerRef: React.RefObject<HTMLElement | null>) {
  const [minWidth, setMinWidth] = useState(280);
  const observerRef = useRef<MutationObserver | null>(null);

  const measure = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;

    const controlsRow = el.querySelector('[data-chat-controls]') as HTMLElement | null;
    if (!controlsRow) return;

    // Temporarily let the row expand to its natural width so we can measure it
    const prev = controlsRow.style.width;
    controlsRow.style.width = 'max-content';
    const natural = controlsRow.scrollWidth + 24;
    controlsRow.style.width = prev;

    setMinWidth(Math.max(240, natural));
  }, [containerRef]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    // Re-measure when children change (vendor/model selection)
    observerRef.current = new MutationObserver(measure);
    observerRef.current.observe(el, { childList: true, subtree: true, characterData: true });
    measure();

    return () => observerRef.current?.disconnect();
  }, [containerRef, measure]);

  return minWidth;
}
