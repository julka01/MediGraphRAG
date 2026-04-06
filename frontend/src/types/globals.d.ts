/* CSS modules */
declare module '*.css' {
  const styles: Record<string, string>;
  export default styles;
}

/* vis-network (loaded via CDN) */
declare namespace vis {
  class DataSet<T = Record<string, unknown>> {
    constructor(data?: T[]);
    add(data: T | T[]): void;
    update(data: T | T[]): void;
    remove(id: string | number | Array<string | number>): void;
    get(): T[];
    get(id: string | number): T | null;
    get(ids: Array<string | number>): T[];
  }

  interface NetworkBody {
    data: {
      nodes: DataSet;
      edges: DataSet;
    };
  }

  interface MoveToOptions {
    position?: { x: number; y: number };
    scale?: number;
    animation?: boolean | { duration?: number; easingFunction?: string };
  }

  interface FitOptions {
    animation?: boolean | { duration?: number; easingFunction?: string };
    nodes?: Array<string | number>;
  }

  interface Canvas {
    frame: {
      canvas: HTMLCanvasElement;
    };
  }

  class Network {
    constructor(
      container: HTMLElement,
      data: { nodes: DataSet; edges: DataSet },
      options?: Record<string, unknown>,
    );
    body: NetworkBody;
    canvas: Canvas;
    destroy(): void;
    on(event: string, callback: (params: Record<string, unknown>) => void): void;
    off(event: string, callback: (params: Record<string, unknown>) => void): void;
    getScale(): number;
    getViewPosition(): { x: number; y: number };
    moveTo(options: MoveToOptions): void;
    fit(options?: FitOptions): void;
    setOptions(options: Record<string, unknown>): void;
    redraw(): void;
  }
}

/* Prism.js (loaded via CDN) */
declare const Prism: {
  highlightElement(el: Element): void;
};
