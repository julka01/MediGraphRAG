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
  vendorLabel: string;
  modelLabel: string;
  vendorHook: UseModelsReturn;
}

export function ModelSelector({ vendorLabel, modelLabel, vendorHook }: ModelSelectorProps) {
  const { vendor, models, selectedModel, loading, setSelectedModel, changeVendor, fetchModels } = vendorHook;

  useEffect(() => {
    fetchModels(vendor);
  }, [vendor, fetchModels]);

  return (
    <>
      {/* Vendor */}
      <div className="relative">
        <select
          value={vendor}
          onChange={(e) => changeVendor(e.target.value)}
          className="select select-bordered select-sm w-full"
        >
          {VENDORS.map((v) => (
            <option key={v.value} value={v.value}>
              {v.label}
            </option>
          ))}
        </select>
        <span className="absolute -top-2 right-3 bg-base-200 px-1 text-[10px] text-base-content/50">
          {vendorLabel}
        </span>
      </div>

      {/* Model */}
      <div className="relative">
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          disabled={loading}
          className="select select-bordered select-sm w-full"
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
        <span className="absolute -top-2 right-3 bg-base-200 px-1 text-[10px] text-base-content/50">
          {modelLabel}
        </span>
      </div>
    </>
  );
}
