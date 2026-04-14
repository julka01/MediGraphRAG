import { renderHook, waitFor } from '@testing-library/react';
import { api } from '../../api';
import { useStartup } from '../useStartup';

vi.mock('../../api', () => ({
  api: {
    checkHealth: vi.fn(),
    fetchModels: vi.fn(),
  },
}));

describe('useStartup', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches health and models in parallel', async () => {
    const mockHealth = { status: 'ok', checks: [] };
    const mockModels = { models: ['model-a', 'model-b'] };

    vi.mocked(api.checkHealth).mockResolvedValue(mockHealth);
    vi.mocked(api.fetchModels).mockResolvedValue(mockModels);

    const { result } = renderHook(() => useStartup(['openai', 'openrouter']));

    expect(result.current.loading).toBe(true);

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(api.checkHealth).toHaveBeenCalledTimes(1);
    expect(api.fetchModels).toHaveBeenCalledTimes(2);
    expect(api.fetchModels).toHaveBeenCalledWith('openai');
    expect(api.fetchModels).toHaveBeenCalledWith('openrouter');

    expect(result.current.health).toEqual(mockHealth);
    expect(result.current.modelsByVendor).toEqual({
      openai: ['model-a', 'model-b'],
      openrouter: ['model-a', 'model-b'],
    });
  });

  it('handles partial failures gracefully', async () => {
    vi.mocked(api.checkHealth).mockResolvedValue({ status: 'ok', checks: [] });
    vi.mocked(api.fetchModels).mockResolvedValue({ models: ['m1'] });

    const { result } = renderHook(() => useStartup(['openai']));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.health).toEqual({ status: 'ok', checks: [] });
    expect(result.current.modelsByVendor).toEqual({ openai: ['m1'] });
  });
});
