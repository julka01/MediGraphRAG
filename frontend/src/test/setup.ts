import '@testing-library/jest-dom';

// Stub vis-network CDN global
const noop = () => {};

const mockNetwork = {
  destroy: noop,
  on: noop,
  off: noop,
  setOptions: noop,
  redraw: noop,
  getScale: () => 1,
  getViewPosition: () => ({ x: 0, y: 0 }),
  fit: noop,
  moveTo: noop,
  body: { data: { nodes: { get: () => null }, edges: { get: () => null } } },
  canvas: { frame: { canvas: document.createElement('canvas') } },
};

globalThis.vis = {
  Network: vi.fn(() => mockNetwork) as unknown as typeof vis.Network,
  DataSet: vi.fn((data: unknown[]) => ({
    get: (id: number) => (data as Array<{ id: number }>).find((d) => d.id === id) ?? null,
    getIds: () => (data as Array<{ id: number }>).map((d) => d.id),
    length: (data as unknown[]).length,
  })) as unknown as typeof vis.DataSet,
} as typeof vis;

// Stub Prism.js CDN global — Prism is declared as `declare const` (not on globalThis),
// so we cast to bypass the type gap between the CDN global declaration and globalThis.
(globalThis as Record<string, unknown>).Prism = {
  highlightElement: noop,
};
