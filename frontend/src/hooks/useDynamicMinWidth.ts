import { useEffect, useState } from 'react';

/**
 * Measures the natural (unshrunk) width of [data-chat-controls] and locks it
 * as the sidebar's minimum width so controls never squeeze into each other.
 *
 * Listens for 'chat-controls-changed' custom events dispatched when dropdown
 * values change, triggering re-measurement.
 */
export function useDynamicMinWidth(containerRef: React.RefObject<HTMLElement | null>) {
  const [minWidth, setMinWidth] = useState(280);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    function measure() {
      // Query fresh each time — React may replace the element
      const controlsRow = container!.querySelector('[data-chat-controls]') as HTMLElement | null;
      if (!controlsRow) return;
      const prev = controlsRow.style.width;
      controlsRow.style.width = 'max-content';
      const natural = controlsRow.scrollWidth + 24;
      controlsRow.style.width = prev;
      setMinWidth(Math.max(240, natural));
    }

    const observer = new ResizeObserver(measure);
    const controlsRow = container.querySelector('[data-chat-controls]') as HTMLElement | null;
    if (controlsRow) observer.observe(controlsRow);

    function handleChange() {
      requestAnimationFrame(measure);
    }
    window.addEventListener('chat-controls-changed', handleChange);

    measure();

    return () => {
      observer.disconnect();
      window.removeEventListener('chat-controls-changed', handleChange);
    };
  }, [containerRef]);

  return minWidth;
}
