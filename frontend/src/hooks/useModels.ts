import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { UseModelsReturn } from '../types/app';

const FALLBACK_MODELS: Record<string, string[]> = {
  openai: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'],
  openrouter: [
    'openai/gpt-oss-120b:free',
    'meta-llama/llama-3.3-8b-instruct:free',
    'deepseek/deepseek-chat-v3.1:free',
    'x-ai/grok-4-fast:free',
  ],
  ollama: [],
  anthropic: ['claude-sonnet-4-20250514', 'claude-haiku-4-20250414'],
  google: ['gemini-2.5-flash', 'gemini-2.5-pro'],
  lmu_lightllm: [],
};

function selectDefault(vendor: string, modelList: string[]): string {
  if (modelList.length === 0) return '';
  if (vendor === 'openrouter' && modelList.includes('openai/gpt-oss-120b:free')) {
    return 'openai/gpt-oss-120b:free';
  }
  return modelList[0];
}

export function useModels(defaultVendor: string, initialModels?: string[]): UseModelsReturn {
  const [vendor, setVendor] = useState(defaultVendor);
  const fallbackInit = initialModels?.length ? initialModels : (FALLBACK_MODELS[defaultVendor] ?? []);
  const [models, setModels] = useState<string[]>(fallbackInit);
  const [selectedModel, setSelectedModel] = useState(() => selectDefault(defaultVendor, fallbackInit));
  const [loading, setLoading] = useState(false);
  const synced = useRef(!!initialModels?.length);

  // Sync initial models when startup data arrives after first render
  useEffect(() => {
    if (!synced.current && initialModels && initialModels.length > 0) {
      synced.current = true;
      setModels(initialModels);
      setSelectedModel(selectDefault(defaultVendor, initialModels));
    }
  }, [initialModels, defaultVendor]);

  const fetchModels = useCallback(
    async (v?: string) => {
      const vendorToFetch = v || vendor;
      setLoading(true);
      try {
        const data = await api.fetchModels(vendorToFetch);
        const modelList = data.models || [];
        if (modelList.length > 0) {
          setModels(modelList);
          setSelectedModel(selectDefault(vendorToFetch, modelList));
        } else {
          const fallback = FALLBACK_MODELS[vendorToFetch] ?? [];
          setModels(fallback);
          setSelectedModel(selectDefault(vendorToFetch, fallback));
        }
      } catch {
        const fallback = FALLBACK_MODELS[vendorToFetch] ?? [];
        setModels(fallback);
        setSelectedModel(selectDefault(vendorToFetch, fallback));
      } finally {
        setLoading(false);
      }
    },
    [vendor],
  );

  const changeVendor = useCallback(
    (newVendor: string) => {
      if (newVendor === vendor) return;
      setVendor(newVendor);
      fetchModels(newVendor);
    },
    [vendor, fetchModels],
  );

  const restoreVendorModel = useCallback(
    async (targetVendor: string, targetModel: string) => {
      if (!Object.keys(FALLBACK_MODELS).includes(targetVendor)) return;
      setVendor(targetVendor);
      setLoading(true);
      try {
        const data = await api.fetchModels(targetVendor);
        const modelList = data.models?.length ? data.models : (FALLBACK_MODELS[targetVendor] ?? []);
        setModels(modelList);
        setSelectedModel(modelList.includes(targetModel) ? targetModel : selectDefault(targetVendor, modelList));
      } catch {
        const fallback = FALLBACK_MODELS[targetVendor] ?? [];
        setModels(fallback);
        setSelectedModel(fallback.includes(targetModel) ? targetModel : selectDefault(targetVendor, fallback));
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  return { vendor, models, selectedModel, loading, setSelectedModel, changeVendor, fetchModels, restoreVendorModel };
}

export function formatModelName(model: string, vendor: string): string {
  if (vendor !== 'openrouter') return model;
  const parts = model.split('/');
  const displayName = parts.length > 1 ? parts[parts.length - 1] : model;
  return displayName.split(':')[0];
}
