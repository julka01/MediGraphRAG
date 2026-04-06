import '@testing-library/jest-dom';

// Node 25 exposes a native `localStorage` stub that lacks .clear/.setItem/.getItem
// unless --localstorage-file is supplied. Override it with a real in-memory implementation
// so that tests using jsdom localStorage work correctly.
const localStorageStore: Record<string, string> = {};
const localStorageMock: Storage = {
  length: 0,
  key: (index: number) => Object.keys(localStorageStore)[index] ?? null,
  getItem: (key: string) => localStorageStore[key] ?? null,
  setItem: (key: string, value: string) => {
    localStorageStore[key] = value;
    (localStorageMock as { length: number }).length = Object.keys(localStorageStore).length;
  },
  removeItem: (key: string) => {
    delete localStorageStore[key];
    (localStorageMock as { length: number }).length = Object.keys(localStorageStore).length;
  },
  clear: () => {
    for (const k of Object.keys(localStorageStore)) delete localStorageStore[k];
    (localStorageMock as { length: number }).length = 0;
  },
};
Object.defineProperty(globalThis, 'localStorage', { value: localStorageMock, writable: true });

// Stub vis-network CDN global
const noop = () => {};

const mockCanvas = document.createElement('canvas');

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
  canvas: { frame: { canvas: mockCanvas } },
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
