import { useEffect } from 'react';
import { formatModelName } from '../../hooks/useModels';
import type { UseModelsReturn } from '../../types/app';
import { FieldsetDropdown } from '../ui/FieldsetDropdown';

const VENDORS = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'openrouter', label: 'OpenRouter' },
  { value: 'lmu_lightllm', label: 'LMU Lightllm' },
  { value: 'ollama', label: 'Ollama' },
  { value: 'anthropic', label: 'Anthropic' },
  { value: 'google', label: 'Google' },
];

interface ModelSelectorProps {
  vendorLabel: string;
  modelLabel: string;
  vendorHook: UseModelsReturn;
}

export function ModelSelector({ vendorLabel, modelLabel, vendorHook }: ModelSelectorProps) {
  const { vendor, models, selectedModel, loading, setSelectedModel, changeVendor, fetchModels } = vendorHook;

  useEffect(() => {
    fetchModels(vendor);
  }, [vendor, fetchModels]);

  const modelOptions = loading
    ? [{ value: '', label: 'Loading models...' }]
    : models.length === 0
      ? [{ value: '', label: 'No models available' }]
      : models.map((m) => ({ value: m, label: formatModelName(m, vendor) }));

  return (
    <>
      <FieldsetDropdown
        label={vendorLabel}
        options={VENDORS}
        value={vendor}
        onChange={changeVendor}
      />
      <FieldsetDropdown
        label={modelLabel}
        options={modelOptions}
        value={selectedModel}
        onChange={setSelectedModel}
        disabled={loading}
      />
    </>
  );
}
