import { useEffect } from 'react';
import { formatModelName } from '../../hooks/useModels';
import type { UseModelsReturn } from '../../types/app';

const VENDORS = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'openrouter', label: 'OpenRouter' },
  { value: 'lmu_lightllm', label: 'LMU Lightllm' },
  { value: 'ollama', label: 'Ollama' },
  { value: 'anthropic', label: 'Anthropic' },
  { value: 'google', label: 'Google' },
] as const;

interface ModelSelectorProps {
  label: string;
  vendorHook: UseModelsReturn;
}

export function ModelSelector({ label, vendorHook }: ModelSelectorProps) {
  const { vendor, models, selectedModel, loading, setSelectedModel, changeVendor, fetchModels } = vendorHook;

  useEffect(() => {
    fetchModels(vendor);
  }, [vendor, fetchModels]);

  return (
    <div className="space-y-2">
      <fieldset className="fieldset">
        <legend className="fieldset-legend text-xs">{label} Vendor</legend>
        <select
          className="select select-bordered select-sm w-full"
          value={vendor}
          onChange={(e) => changeVendor(e.target.value)}
        >
          {VENDORS.map((v) => (
            <option key={v.value} value={v.value}>
              {v.label}
            </option>
          ))}
        </select>
      </fieldset>
      <fieldset className="fieldset">
        <legend className="fieldset-legend text-xs">{label} Model</legend>
        <select
          className="select select-bordered select-sm w-full"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          disabled={loading}
        >
          {loading ? (
            <option value="">Loading models...</option>
          ) : models.length === 0 ? (
            <option value="">No models available</option>
          ) : (
            models.map((m) => (
              <option key={m} value={m}>
                {formatModelName(m, vendor)}
              </option>
            ))
          )}
        </select>
      </fieldset>
    </div>
  );
}
