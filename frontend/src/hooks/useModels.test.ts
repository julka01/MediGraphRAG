import { act, renderHook } from '@testing-library/react';
import { api } from '../api';
import { formatModelName, useModels } from './useModels';

vi.mock('../api', () => ({
  api: {
    fetchModels: vi.fn(),
  },
}));

const mockFetchModels = vi.mocked(api.fetchModels);

describe('useModels', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('initializes with default vendor and empty models', () => {
    const { result } = renderHook(() => useModels('openai'));
    expect(result.current.vendor).toBe('openai');
    expect(result.current.models).toEqual([]);
    expect(result.current.selectedModel).toBe('');
    expect(result.current.loading).toBe(false);
  });

  it('fetches models and auto-selects the first one', async () => {
    mockFetchModels.mockResolvedValue({ models: ['gpt-4', 'gpt-3.5-turbo'] });
    const { result } = renderHook(() => useModels('openai'));

    await act(async () => {
      await result.current.fetchModels();
    });

    expect(mockFetchModels).toHaveBeenCalledWith('openai');
    expect(result.current.models).toEqual(['gpt-4', 'gpt-3.5-turbo']);
    expect(result.current.selectedModel).toBe('gpt-4');
    expect(result.current.loading).toBe(false);
  });

  it('auto-selects openrouter free model when available', async () => {
    mockFetchModels.mockResolvedValue({
      models: ['meta/llama-3', 'openai/gpt-oss-120b:free', 'google/gemini'],
    });
    const { result } = renderHook(() => useModels('openrouter'));

    await act(async () => {
      await result.current.fetchModels('openrouter');
    });

    expect(result.current.selectedModel).toBe('openai/gpt-oss-120b:free');
  });

  it('selects first model when openrouter free model is not in list', async () => {
    mockFetchModels.mockResolvedValue({
      models: ['meta/llama-3', 'google/gemini'],
    });
    const { result } = renderHook(() => useModels('openrouter'));

    await act(async () => {
      await result.current.fetchModels('openrouter');
    });

    expect(result.current.selectedModel).toBe('meta/llama-3');
  });

  it('sets selectedModel to empty string on empty model list', async () => {
    mockFetchModels.mockResolvedValue({ models: [] });
    const { result } = renderHook(() => useModels('openai'));

    await act(async () => {
      await result.current.fetchModels();
    });

    expect(result.current.models).toEqual([]);
    expect(result.current.selectedModel).toBe('');
  });

  it('handles API error gracefully', async () => {
    mockFetchModels.mockRejectedValue(new Error('Network error'));
    const { result } = renderHook(() => useModels('openai'));

    await act(async () => {
      await result.current.fetchModels();
    });

    expect(result.current.models).toEqual([]);
    expect(result.current.selectedModel).toBe('');
    expect(result.current.loading).toBe(false);
  });

  it('changeVendor updates vendor and triggers fetch', async () => {
    mockFetchModels.mockResolvedValue({ models: ['llama-3'] });
    const { result } = renderHook(() => useModels('openai'));

    await act(async () => {
      result.current.changeVendor('ollama');
    });

    expect(result.current.vendor).toBe('ollama');
    expect(mockFetchModels).toHaveBeenCalledWith('ollama');
  });
});

describe('formatModelName', () => {
  it('strips vendor prefix for openrouter models', () => {
    expect(formatModelName('openai/gpt-4o:latest', 'openrouter')).toBe('gpt-4o');
  });

  it('strips version suffix for openrouter models', () => {
    expect(formatModelName('meta/llama-3:free', 'openrouter')).toBe('llama-3');
  });

  it('handles model without vendor prefix on openrouter', () => {
    expect(formatModelName('gpt-4o:latest', 'openrouter')).toBe('gpt-4o');
  });

  it('returns model as-is for non-openrouter vendors', () => {
    expect(formatModelName('gpt-4o', 'openai')).toBe('gpt-4o');
    expect(formatModelName('llama-3', 'ollama')).toBe('llama-3');
  });
});
