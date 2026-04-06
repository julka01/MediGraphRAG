import { act, renderHook } from '@testing-library/react';
import type { ReactNode } from 'react';
import { GraphProvider, useGraphState } from '../GraphContext';

function wrapper({ children }: { children: ReactNode }) {
  return <GraphProvider>{children}</GraphProvider>;
}

describe('GraphContext', () => {
  it('provides initial state', () => {
    const { result } = renderHook(() => useGraphState(), { wrapper });
    expect(result.current.state.graphData).toBeNull();
    expect(result.current.state.physicsEnabled).toBe(true);
    expect(result.current.state.highlightedNodes.size).toBe(0);
  });

  it('handles SET_GRAPH_DATA', () => {
    const { result } = renderHook(() => useGraphState(), { wrapper });
    const data = { nodes: [{ id: '1' }], relationships: [] };
    act(() => {
      result.current.dispatch({ type: 'SET_GRAPH_DATA', data });
    });
    expect(result.current.state.graphData).toEqual(data);
  });

  it('handles TOGGLE_PHYSICS', () => {
    const { result } = renderHook(() => useGraphState(), { wrapper });
    act(() => {
      result.current.dispatch({ type: 'TOGGLE_PHYSICS' });
    });
    expect(result.current.state.physicsEnabled).toBe(false);
  });

  it('handles SET_HIGHLIGHTED_NODES and CLEAR_HIGHLIGHTED_NODES', () => {
    const { result } = renderHook(() => useGraphState(), { wrapper });
    act(() => {
      result.current.dispatch({ type: 'SET_HIGHLIGHTED_NODES', nodes: ['a', 'b'] });
    });
    expect(result.current.state.highlightedNodes).toEqual(new Set(['a', 'b']));
    act(() => {
      result.current.dispatch({ type: 'CLEAR_HIGHLIGHTED_NODES' });
    });
    expect(result.current.state.highlightedNodes.size).toBe(0);
  });
});
