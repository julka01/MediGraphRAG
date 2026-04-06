import { api } from '../api.js';

export async function updateModelDropdown(vendorSelectId, modelSelectId) {
  const vendorSelect = document.getElementById(vendorSelectId);
  const modelSelect = document.getElementById(modelSelectId);
  const vendor = vendorSelect.value;

  // Show loading indicator
  modelSelect.innerHTML = '<option value="">Loading models...</option>';

  try {
    const data = await api.fetchModels(vendor);
    modelSelect.innerHTML = '';

    try {
      data.models.forEach(model => {
        const option = document.createElement('option');
        option.value = model;

        // Map model names to more user-friendly display names
        let displayName = model;
        if (vendor === 'openrouter') {
          // Extract the last part after the last slash
          const parts = model.split('/');
          if (parts.length > 1) {
            displayName = parts[parts.length - 1];
          }
          // Remove any suffixes after colon
          displayName = displayName.split(':')[0];
        }

        option.textContent = displayName;
        modelSelect.appendChild(option);
      });

      // Preselect first model
      if (data.models.length > 0) {
        if (vendor === 'openrouter' && data.models.includes('openai/gpt-oss-120b:free')) {
          modelSelect.value = 'openai/gpt-oss-120b:free';
        } else {
          modelSelect.value = data.models[0];
        }
      } else {
        modelSelect.innerHTML = '<option value="">No models available</option>';
      }
    } catch (error) {
      console.error('Error processing models:', error);
      modelSelect.innerHTML = '<option value="">Error processing models</option>';
    }
  } catch (error) {
    console.error('Error fetching models:', error);
    modelSelect.innerHTML = '<option value="">Error loading models</option>';
  }
}

export function initModels() {
  updateModelDropdown('kg-provider', 'kg-model');
  updateModelDropdown('rag-vendor', 'rag-model');

  document.getElementById('kg-provider').addEventListener('change', function () {
    updateModelDropdown('kg-provider', 'kg-model');
  });

  document.getElementById('rag-vendor').addEventListener('change', function () {
    updateModelDropdown('rag-vendor', 'rag-model');
  });
}
