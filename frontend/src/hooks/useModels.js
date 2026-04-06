import { useState, useCallback } from 'react';
import { api } from '../api';

export function useModels(defaultVendor) {
  const [vendor, setVendor] = useState(defaultVendor);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [loading, setLoading] = useState(false);

  const fetchModels = useCallback(async (v) => {
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
  }, [vendor]);

  const changeVendor = useCallback((newVendor) => {
    setVendor(newVendor);
    fetchModels(newVendor);
  }, [fetchModels]);

  return { vendor, models, selectedModel, loading, setSelectedModel, changeVendor, fetchModels };
}

export function formatModelName(model, vendor) {
  if (vendor !== 'openrouter') return model;
  const parts = model.split('/');
  let displayName = parts.length > 1 ? parts[parts.length - 1] : model;
  return displayName.split(':')[0];
}
