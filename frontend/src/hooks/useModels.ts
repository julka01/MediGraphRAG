import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { UseModelsReturn } from '../types/app';

function selectDefault(vendor: string, modelList: string[]): string {
  if (modelList.length === 0) return '';
  if (vendor === 'openrouter' && modelList.includes('openai/gpt-oss-120b:free')) {
    return 'openai/gpt-oss-120b:free';
  }
  return modelList[0];
}

export function useModels(defaultVendor: string, initialModels?: string[]): UseModelsReturn {
  const [vendor, setVendor] = useState(defaultVendor);
  const [models, setModels] = useState<string[]>(initialModels || []);
  const [selectedModel, setSelectedModel] = useState(() => selectDefault(defaultVendor, initialModels || []));
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
        setModels(modelList);
        setSelectedModel(selectDefault(vendorToFetch, modelList));
      } catch {
        setModels([]);
        setSelectedModel('');
      } finally {
        setLoading(false);
      }
    },
    [vendor],
  );

  const changeVendor = useCallback(
    (newVendor: string) => {
      setVendor(newVendor);
      fetchModels(newVendor);
    },
    [fetchModels],
  );

  return { vendor, models, selectedModel, loading, setSelectedModel, changeVendor, fetchModels };
}

export function formatModelName(model: string, vendor: string): string {
  if (vendor !== 'openrouter') return model;
  const parts = model.split('/');
  const displayName = parts.length > 1 ? parts[parts.length - 1] : model;
  return displayName.split(':')[0];
}
