import { act, renderHook } from '@testing-library/react';
import type { ReactNode } from 'react';
import { KGProvider, useKG } from '../KGContext';

function wrapper({ children }: { children: ReactNode }) {
  return <KGProvider>{children}</KGProvider>;
}

describe('KGContext', () => {
  it('provides initial state', () => {
    const { result } = renderHook(() => useKG(), { wrapper });
    expect(result.current.state.currentKGId).toBeNull();
    expect(result.current.state.currentKGName).toBeNull();
    expect(result.current.state.kgList).toEqual([]);
  });

  it('handles SET_KG', () => {
    const { result } = renderHook(() => useKG(), { wrapper });
    act(() => {
      result.current.dispatch({ type: 'SET_KG', kgId: '123', kgName: 'test' });
    });
    expect(result.current.state.currentKGId).toBe('123');
    expect(result.current.state.currentKGName).toBe('test');
  });

  it('handles CLEAR_KG', () => {
    const { result } = renderHook(() => useKG(), { wrapper });
    act(() => {
      result.current.dispatch({ type: 'SET_KG', kgId: '123', kgName: 'test' });
    });
    act(() => {
      result.current.dispatch({ type: 'CLEAR_KG' });
    });
    expect(result.current.state.currentKGId).toBeNull();
    expect(result.current.state.currentKGName).toBeNull();
  });
});
