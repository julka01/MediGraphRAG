import { act, renderHook } from '@testing-library/react';
import type { ReactNode } from 'react';
import { UIProvider, useUI } from '../UIContext';

function wrapper({ children }: { children: ReactNode }) {
  return <UIProvider>{children}</UIProvider>;
}

describe('UIContext', () => {
  it('provides initial state', () => {
    const { result } = renderHook(() => useUI(), { wrapper });
    expect(result.current.state.layout).toBe('split');
    expect(result.current.state.sidebarCollapsed).toBe(false);
    expect(result.current.state.notification).toBeNull();
  });

  it('handles SET_LAYOUT', () => {
    const { result } = renderHook(() => useUI(), { wrapper });
    act(() => {
      result.current.dispatch({ type: 'SET_LAYOUT', layout: 'graph-only' });
    });
    expect(result.current.state.layout).toBe('graph-only');
  });

  it('handles TOGGLE_SIDEBAR', () => {
    const { result } = renderHook(() => useUI(), { wrapper });
    act(() => {
      result.current.dispatch({ type: 'TOGGLE_SIDEBAR' });
    });
    expect(result.current.state.sidebarCollapsed).toBe(true);
  });

  it('handles SHOW_NOTIFICATION and CLEAR_NOTIFICATION', () => {
    const { result } = renderHook(() => useUI(), { wrapper });
    act(() => {
      result.current.dispatch({ type: 'SHOW_NOTIFICATION', notifType: 'success', message: 'done' });
    });
    expect(result.current.state.notification).toEqual({ type: 'success', message: 'done' });
    act(() => {
      result.current.dispatch({ type: 'CLEAR_NOTIFICATION' });
    });
    expect(result.current.state.notification).toBeNull();
  });
});
