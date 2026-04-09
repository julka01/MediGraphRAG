import { ArrowUpTrayIcon, CheckIcon, ChevronDownIcon } from '@heroicons/react/24/outline';
import { useRef, useState } from 'react';
import { api } from '../../api';
import { useApp } from '../../context/AppContext';
import type { UseModelsReturn } from '../../types/app';
import { safeSet } from '../../utils/storage';
import { showError, showSuccess } from '../ui/Notifications';
import { ModelSelector } from './ModelSelector';

interface KGPanelProps {
  kgModelHook: UseModelsReturn;
  onLoadKG: (loadMode: string, nodeLimit: number, kgFilter: string) => void;
  onProgressStart: () => void;
  onProgressStop: () => void;
}

export function KGPanel({ kgModelHook, onLoadKG, onProgressStart, onProgressStop }: KGPanelProps) {
  const { state, dispatch } = useApp();
  const fileRef = useRef<HTMLInputElement>(null);
  const ontologyRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [ontologyFile, setOntologyFile] = useState<File | null>(null);
  const [kgName, setKgName] = useState('');
  const [embeddingModel, setEmbeddingModel] = useState('sentence_transformers');
  const [maxChunks, setMaxChunks] = useState(20);
  const [creating, setCreating] = useState(false);

  // Load section state
  const [loadMode, setLoadMode] = useState('limited');
  const [nodeLimit, setNodeLimit] = useState(1000);
  const [kgFilter, setKgFilter] = useState('');

  // KG name placeholder: show loaded KG name when user hasn't typed anything
  const kgNamePlaceholder = !kgName && state.currentKGName ? state.currentKGName : 'optional';

  const canCreate =
    file !== null &&
    ontologyFile !== null &&
    Boolean(kgModelHook.vendor) &&
    Boolean(kgModelHook.selectedModel) &&
    Boolean(embeddingModel);

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

  // Drag-and-drop handlers for file upload zones
  const handleFileDrop = (e: React.DragEvent<HTMLButtonElement>) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) setFile(dropped);
  };

  const handleOntologyDrop = (e: React.DragEvent<HTMLButtonElement>) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files?.[0];
    if (!dropped) return;
    if (!dropped.name.endsWith('.json') && !dropped.name.endsWith('.owl')) {
      showError(dispatch, 'Only JSON and OWL ontology files are supported');
      return;
    }
    setOntologyFile(dropped);
  };

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

  const handleDelete = async () => {
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

  const handleLoadKG = () => {
    onLoadKG(loadMode, nodeLimit, kgFilter);
  };

  return (
    <div className="flex flex-col gap-3">
      {/* ── Build Section ─────────────────────────────────── */}

      {/* File upload drop zone */}
      <input ref={fileRef} type="file" accept=".pdf,.txt,.json,.csv" className="hidden" onChange={handleFileChange} />
      <button
        type="button"
        className={[
          'flex w-full items-center gap-2 rounded-lg border px-3 py-2 text-left text-sm transition-colors',
          file
            ? 'border-success/60 bg-success/5 text-base-content'
            : 'border-dashed border-base-300 text-base-content/50 hover:border-base-content/30 hover:text-base-content/70',
        ].join(' ')}
        onClick={() => fileRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleFileDrop}
      >
        {file ? (
          <CheckIcon className="size-4 shrink-0 text-success" />
        ) : (
          <ArrowUpTrayIcon className="size-4 shrink-0" />
        )}
        <span className="truncate">{file ? `File: ${file.name}` : 'Drop or select file'}</span>
      </button>

      {/* Ontology upload drop zone */}
      <input ref={ontologyRef} type="file" accept=".json,.owl" className="hidden" onChange={handleOntologyChange} />
      <button
        type="button"
        className={[
          'flex w-full items-center gap-2 rounded-lg border px-3 py-2 text-left text-sm transition-colors',
          ontologyFile
            ? 'border-success/60 bg-success/5 text-base-content'
            : 'border-dashed border-base-300 text-base-content/50 hover:border-base-content/30 hover:text-base-content/70',
        ].join(' ')}
        onClick={() => ontologyRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleOntologyDrop}
      >
        {ontologyFile ? (
          <CheckIcon className="size-4 shrink-0 text-success" />
        ) : (
          <ArrowUpTrayIcon className="size-4 shrink-0" />
        )}
        <span className="truncate">{ontologyFile ? `Ontology: ${ontologyFile.name}` : 'Drop or select ontology'}</span>
      </button>

      {/* KG Vendor + KG Model */}
      <ModelSelector vendorLabel="KG Vendor" modelLabel="KG Model" vendorHook={kgModelHook} />

      {/* Embedding Model */}
      <div className="relative">
        <select
          className="select select-bordered select-sm w-full appearance-none pr-8 focus:outline-none focus:ring-1 focus:ring-primary/30"
          value={embeddingModel}
          onChange={(e) => setEmbeddingModel(e.target.value)}
        >
          <option value="sentence_transformers">Sentence Transformers</option>
          <option value="openai">OpenAI Embeddings</option>
          <option value="vertexai">Google Vertex AI</option>
          <option value="titan">AWS Titan</option>
        </select>
        <ChevronDownIcon className="pointer-events-none absolute right-2 top-1/2 size-3.5 -translate-y-1/2 text-base-content/40" />
        <span className="absolute -top-2 right-3 bg-base-200 px-1 text-2xs text-base-content/50">Embedding Model</span>
      </div>

      {/* Max Chunks */}
      <div className="relative">
        <input
          type="number"
          className="input input-bordered input-sm w-full focus:outline-none focus:ring-1 focus:ring-primary/30"
          value={maxChunks}
          onChange={(e) => setMaxChunks(Number(e.target.value))}
          min={1}
          max={100}
        />
        <span className="absolute -top-2 right-3 bg-base-200 px-1 text-2xs text-base-content/50">Max Chunks</span>
      </div>

      {/* KG Name */}
      <div className="relative">
        <input
          type="text"
          className="input input-bordered input-sm w-full focus:outline-none focus:ring-1 focus:ring-primary/30"
          placeholder={kgNamePlaceholder}
          value={kgName}
          onChange={(e) => setKgName(e.target.value)}
        />
        <span className="absolute -top-2 right-3 bg-base-200 px-1 text-2xs text-base-content/50">KG Name</span>
      </div>

      {/* Create KG */}
      <button
        type="button"
        className="btn btn-primary btn-sm w-full"
        onClick={handleCreate}
        disabled={creating || !canCreate}
      >
        {creating && <span className="loading loading-spinner loading-sm" />}
        {creating ? 'Creating KG...' : 'Create KG'}
      </button>

      {/* ── Decorative Separator ──────────────────────────── */}
      <div className="flex items-center gap-3 my-2">
        <div className="h-px flex-1 bg-gradient-to-r from-transparent to-base-300" />
        <div className="size-1 rounded-full bg-base-300" />
        <div className="h-px flex-1 bg-gradient-to-l from-transparent to-base-300" />
      </div>

      {/* ── Load Section ──────────────────────────────────── */}

      {/* Import Options */}
      <div className="relative">
        <select
          className="select select-bordered select-sm w-full appearance-none pr-8 focus:outline-none focus:ring-1 focus:ring-primary/30"
          value={loadMode}
          onChange={(e) => setLoadMode(e.target.value)}
        >
          <option value="limited">Limited (1000 nodes max)</option>
          <option value="sample">Smart Sample</option>
          <option value="complete">Complete Import</option>
        </select>
        <ChevronDownIcon className="pointer-events-none absolute right-2 top-1/2 size-3.5 -translate-y-1/2 text-base-content/40" />
        <span className="absolute -top-2 right-3 bg-base-200 px-1 text-2xs text-base-content/50">Import Options</span>
      </div>

      {/* Node Limit */}
      <div className="relative">
        <input
          type="number"
          className="input input-bordered input-sm w-full focus:outline-none focus:ring-1 focus:ring-primary/30"
          value={nodeLimit}
          onChange={(e) => setNodeLimit(Number(e.target.value))}
          min={1}
        />
        <span className="absolute -top-2 right-3 bg-base-200 px-1 text-2xs text-base-content/50">Node Limit</span>
      </div>

      {/* KG Name Filter */}
      <div className="relative">
        <select
          className="select select-bordered select-sm w-full appearance-none pr-8 focus:outline-none focus:ring-1 focus:ring-primary/30"
          value={kgFilter}
          onChange={(e) => setKgFilter(e.target.value)}
        >
          <option value="">All KGs</option>
          {state.kgList.map((kg) => (
            <option key={kg.name} value={kg.name}>
              {kg.name}
            </option>
          ))}
        </select>
        <ChevronDownIcon className="pointer-events-none absolute right-2 top-1/2 size-3.5 -translate-y-1/2 text-base-content/40" />
        <span className="absolute -top-2 right-3 bg-base-200 px-1 text-2xs text-base-content/50">KG Name</span>
      </div>

      {/* Load KG + Delete KG */}
      <div className="flex gap-2">
        <button type="button" className="btn btn-primary btn-sm flex-1" onClick={handleLoadKG}>
          Load KG
        </button>
        <button
          type="button"
          className={[
            'btn btn-sm flex-1',
            state.currentKGId
              ? 'btn-outline border-error/50 text-error hover:bg-error hover:text-error-content'
              : 'btn-outline text-base-content/30 pointer-events-none',
          ].join(' ')}
          onClick={handleDelete}
          disabled={!state.currentKGId}
        >
          Delete KG
        </button>
      </div>
    </div>
  );
}
