import { useEffect, useRef, useState } from 'react';

export function useDynamicMinWidth(containerRef: React.RefObject<HTMLElement | null>) {
  const [minWidth, setMinWidth] = useState(280);
  const observerRef = useRef<ResizeObserver | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const measure = () => {
      const controlsRow = el.querySelector('[data-chat-controls]');
      if (controlsRow) {
        const width = (controlsRow as HTMLElement).scrollWidth + 24;
        setMinWidth(Math.max(240, width));
      }
    };

    observerRef.current = new ResizeObserver(measure);
    observerRef.current.observe(el);
    measure();

    return () => observerRef.current?.disconnect();
  }, [containerRef]);

  return minWidth;
}
