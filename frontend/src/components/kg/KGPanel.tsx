import { CheckIcon } from '@heroicons/react/24/outline';
import { useEffect, useRef, useState } from 'react';
import { api } from '../../api';
import { useApp } from '../../context/AppContext';
import type { UseModelsReturn } from '../../types/app';
import { safeSet } from '../../utils/storage';
import { showError, showSuccess } from '../ui/Notifications';
import { ModelSelector } from './ModelSelector';

interface KGPanelProps {
  kgModelHook: UseModelsReturn;
  onNeo4jOpen: () => void;
  onProgressStart: () => void;
  onProgressStop: () => void;
}

export function KGPanel({ kgModelHook, onNeo4jOpen, onProgressStart, onProgressStop }: KGPanelProps) {
  const { state, dispatch } = useApp();
  const fileRef = useRef<HTMLInputElement>(null);
  const ontologyRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [ontologyFile, setOntologyFile] = useState<File | null>(null);
  const [kgName, setKgName] = useState('');
  const [embeddingModel, setEmbeddingModel] = useState('sentence_transformers');
  const [maxChunks, setMaxChunks] = useState(20);
  const [creating, setCreating] = useState(false);

  // Sync KG name from state when a KG is loaded from Neo4j
  useEffect(() => {
    if (state.currentKGName) {
      setKgName(state.currentKGName);
    }
  }, [state.currentKGName]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) setFile(selected);
  };

  const handleOntologyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (!selected) return;
    if (!selected.name.endsWith('.json') && !selected.name.endsWith('.owl')) {
      showError(dispatch, 'Only JSON and OWL ontology files are supported');
      e.target.value = '';
      return;
    }
    setOntologyFile(selected);
  };

  const canCreate = file && kgModelHook.vendor && kgModelHook.selectedModel && embeddingModel;

  const handleCreate = async () => {
    if (!file) {
      showError(dispatch, 'Please select a file first');
      return;
    }
    setCreating(true);
    onProgressStart();
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('provider', kgModelHook.vendor);
      formData.append('model', kgModelHook.selectedModel);
      formData.append('embedding_model', embeddingModel);
      formData.append('max_chunks', String(maxChunks));
      if (kgName) formData.append('kg_name', kgName);
      if (ontologyFile) formData.append('ontology_file', ontologyFile);
      const result = await api.createKG(formData);
      onProgressStop();
      dispatch({ type: 'SET_KG', kgId: result.kg_id, kgName: result.kg_name || null });
      if (result.kg_name) safeSet('currentKGName', result.kg_name);
      if (result.graph_data) dispatch({ type: 'SET_GRAPH_DATA', data: result.graph_data });
      setFile(null);
      if (fileRef.current) fileRef.current.value = '';
      showSuccess(
        dispatch,
        `KG created: ${result.graph_data?.nodes?.length || 0} nodes, ${result.graph_data?.relationships?.length || 0} edges`,
      );
    } catch (error) {
      onProgressStop();
      const msg = error instanceof Error ? error.message : String(error);
      showError(dispatch, `KG creation failed: ${msg}`);
    } finally {
      setCreating(false);
    }
  };

  const handleClear = async () => {
    if (!confirm('WARNING: This will permanently delete ALL nodes and relationships from Neo4j. Continue?')) return;
    try {
      const result = await api.clearKG();
      dispatch({ type: 'CLEAR_KG' });
      localStorage.removeItem('currentKGName');
      showSuccess(dispatch, result.message || 'KG cleared');
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      showError(dispatch, `Failed to clear KG: ${msg}`);
    }
  };

  return (
    <div className="flex flex-col gap-3">
      {/* 1. Load KG + Clear KG */}
      <div className="flex gap-2">
        <button type="button" className="btn btn-primary btn-sm flex-1" onClick={() => onNeo4jOpen()}>
          Load KG
        </button>
        <button type="button" className="btn btn-ghost btn-sm text-error" onClick={handleClear}>
          Clear KG
        </button>
      </div>

      {/* 2. Select File */}
      <input ref={fileRef} type="file" accept=".pdf,.txt,.json,.csv" className="hidden" onChange={handleFileChange} />
      <button
        type="button"
        className="btn btn-outline btn-sm w-full justify-between"
        onClick={() => fileRef.current?.click()}
      >
        <span className="truncate">{file ? file.name : 'Select File'}</span>
        {file && <CheckIcon className="size-4 text-success shrink-0" />}
      </button>

      {/* 3. Select Ontology */}
      <input ref={ontologyRef} type="file" accept=".json,.owl" className="hidden" onChange={handleOntologyChange} />
      <button
        type="button"
        className="btn btn-outline btn-sm w-full justify-between"
        onClick={() => ontologyRef.current?.click()}
      >
        <span className="truncate">{ontologyFile ? ontologyFile.name : 'Select Ontology'}</span>
        {ontologyFile && <CheckIcon className="size-4 text-success shrink-0" />}
      </button>

      {/* 4. KG Vendor + KG Model */}
      <ModelSelector vendorLabel="KG Vendor" modelLabel="KG Model" vendorHook={kgModelHook} />

      {/* 5. Embedding Model */}
      <div className="relative">
        <select
          className="select select-bordered select-sm w-full"
          value={embeddingModel}
          onChange={(e) => setEmbeddingModel(e.target.value)}
        >
          <option value="sentence_transformers">Sentence Transformers</option>
          <option value="openai">OpenAI Embeddings</option>
          <option value="vertexai">Google Vertex AI</option>
          <option value="titan">AWS Titan</option>
        </select>
        <span className="absolute -top-2 right-3 bg-base-200 px-1 text-2xs text-base-content/50">
          Embedding Model
        </span>
      </div>

      {/* 6. Max Chunks per Report */}
      <div className="relative">
        <input
          type="number"
          className="input input-bordered input-sm w-full"
          value={maxChunks}
          onChange={(e) => setMaxChunks(Number(e.target.value))}
          min={1}
          max={100}
        />
        <span className="absolute -top-2 right-3 bg-base-200 px-1 text-2xs text-base-content/50">
          Max Chunks
        </span>
      </div>

      {/* 7. KG Name */}
      <div className="relative">
        <input
          type="text"
          className="input input-bordered input-sm w-full"
          placeholder="optional"
          value={kgName}
          onChange={(e) => setKgName(e.target.value)}
        />
        <span className="absolute -top-2 right-3 bg-base-200 px-1 text-2xs text-base-content/50">
          KG Name
        </span>
      </div>

      {/* 8. Create KG */}
      <button
        type="button"
        className="btn btn-primary btn-sm w-full"
        onClick={handleCreate}
        disabled={creating || !canCreate}
      >
        {creating ? <span className="loading loading-spinner loading-sm" /> : null}
        {creating ? 'Creating KG...' : 'Create KG'}
      </button>
    </div>
  );
}
