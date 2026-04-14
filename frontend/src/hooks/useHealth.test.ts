import { act, renderHook } from '@testing-library/react';
import { api } from '../api';
import { useHealth } from './useHealth';

vi.mock('../api', () => ({
  api: {
    checkHealth: vi.fn(),
  },
}));

const mockCheckHealth = vi.mocked(api.checkHealth);

describe('useHealth', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns initial state with unknown level', () => {
    const { result } = renderHook(() => useHealth());
    expect(result.current.status.level).toBe('unknown');
    expect(result.current.status.tip).toBe('Checking system health…');
    expect(result.current.checking).toBe(false);
  });

  it('returns ok level for ok status', async () => {
    mockCheckHealth.mockResolvedValue({ status: 'ok', checks: [] });
    const { result } = renderHook(() => useHealth());

    await act(async () => {
      await result.current.checkHealth();
    });

    expect(result.current.status.level).toBe('ok');
    expect(result.current.status.tip).toContain('OK');
    expect(result.current.checking).toBe(false);
  });

  it('returns warn level and lists warned checks', async () => {
    mockCheckHealth.mockResolvedValue({
      status: 'warn',
      checks: [
        { check: 'neo4j', status: 'warn' },
        { check: 'ollama', status: 'ok' },
      ],
    });
    const { result } = renderHook(() => useHealth());

    await act(async () => {
      await result.current.checkHealth();
    });

    expect(result.current.status.level).toBe('warn');
    expect(result.current.status.tip).toContain('WARN');
    expect(result.current.status.tip).toContain('neo4j');
  });

  it('returns fail level and lists failed checks', async () => {
    mockCheckHealth.mockResolvedValue({
      status: 'fail',
      checks: [
        { check: 'ollama', status: 'fail' },
        { check: 'neo4j', status: 'ok' },
      ],
    });
    const { result } = renderHook(() => useHealth());

    await act(async () => {
      await result.current.checkHealth();
    });

    expect(result.current.status.level).toBe('fail');
    expect(result.current.status.tip).toContain('FAIL');
    expect(result.current.status.tip).toContain('ollama');
  });

  it('returns fail level with unreachable message on API failure', async () => {
    mockCheckHealth.mockRejectedValue(new Error('Network error'));
    const { result } = renderHook(() => useHealth());

    await act(async () => {
      await result.current.checkHealth();
    });

    expect(result.current.status.level).toBe('fail');
    expect(result.current.status.tip).toContain('unreachable');
    expect(result.current.checking).toBe(false);
  });

  it('handles missing checks array gracefully', async () => {
    mockCheckHealth.mockResolvedValue({ status: 'ok' });
    const { result } = renderHook(() => useHealth());

    await act(async () => {
      await result.current.checkHealth();
    });

    expect(result.current.status.level).toBe('ok');
  });
});
