import { useCallback, useState } from 'react';
import { api } from '../api';
import type { UseModelsReturn } from '../types/app';

export function useModels(defaultVendor: string, initialModels?: string[]): UseModelsReturn {
  const [vendor, setVendor] = useState(defaultVendor);
  const [models, setModels] = useState<string[]>(initialModels || []);
  const [selectedModel, setSelectedModel] = useState(() => {
    if (!initialModels || initialModels.length === 0) return '';
    if (defaultVendor === 'openrouter' && initialModels.includes('openai/gpt-oss-120b:free')) {
      return 'openai/gpt-oss-120b:free';
    }
    return initialModels[0];
  });
  const [loading, setLoading] = useState(false);

  const fetchModels = useCallback(
    async (v?: string) => {
      const vendorToFetch = v || vendor;
      setLoading(true);
      try {
        const data = await api.fetchModels(vendorToFetch);
        const modelList = data.models || [];
        setModels(modelList);

        if (vendorToFetch === 'openrouter' && modelList.includes('openai/gpt-oss-120b:free')) {
          setSelectedModel('openai/gpt-oss-120b:free');
        } else if (modelList.length > 0) {
          setSelectedModel(modelList[0]);
        } else {
          setSelectedModel('');
        }
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
