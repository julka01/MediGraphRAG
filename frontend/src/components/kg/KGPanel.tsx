import { ArrowUpTrayIcon, CheckIcon } from '@heroicons/react/24/outline';
import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../../api';
import { useApp } from '../../context/AppContext';
import type { KGSettings, UseModelsReturn } from '../../types/app';
import { safeSet } from '../../utils/storage';
import { FieldsetDropdown } from '../ui/FieldsetDropdown';
import { showError, showSuccess } from '../ui/Notifications';
import { ModelSelector } from './ModelSelector';

interface KGPanelProps {
  kgModelHook: UseModelsReturn;
  onLoadKG: (loadMode: string, nodeLimit: number, kgFilter: string) => void;
  onProgressStart: () => void;
  onProgressStop: () => void;
  loadedKGSettings?: KGSettings | null;
}

export function KGPanel({ kgModelHook, onLoadKG, onProgressStart, onProgressStop, loadedKGSettings }: KGPanelProps) {
  const { state, dispatch } = useApp();
  const fileRef = useRef<HTMLInputElement>(null);
  const ontologyRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [ontologyFile, setOntologyFile] = useState<File | null>(null);
  const [kgName, setKgName] = useState('');
  const [embeddingModel, setEmbeddingModel] = useState('sentence_transformers');
  const [maxChunks, setMaxChunks] = useState('20');
  const [creating, setCreating] = useState(false);

  // Load section state
  const [importMode, setImportMode] = useState('limited-1000');
  const [kgFilter, setKgFilter] = useState('');

  // Apply settings from loaded KG
  const EMBEDDING_OPTIONS = ['sentence_transformers', 'openai', 'vertexai', 'titan'];
  useEffect(() => {
    if (!loadedKGSettings) return;
    if (loadedKGSettings.embeddingModel && EMBEDDING_OPTIONS.includes(loadedKGSettings.embeddingModel)) {
      setEmbeddingModel(loadedKGSettings.embeddingModel);
    }
    if (loadedKGSettings.maxChunks != null) {
      setMaxChunks(String(loadedKGSettings.maxChunks));
    }
  }, [loadedKGSettings]); // eslint-disable-line react-hooks/exhaustive-deps -- EMBEDDING_OPTIONS is stable

  // KG name placeholder: show loaded KG name when user hasn't typed anything
  const kgNamePlaceholder = !kgName && state.currentKGName ? state.currentKGName : 'optional';

  const canCreate =
    file !== null &&
    ontologyFile !== null &&
    maxChunks !== '' &&
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
    if (!file) return;
    setCreating(true);
    onProgressStart();
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('provider', kgModelHook.vendor);
      formData.append('model', kgModelHook.selectedModel);
      formData.append('embedding_model', embeddingModel);
      formData.append('max_chunks', maxChunks);
      if (kgName) formData.append('kg_name', kgName);
      if (ontologyFile) formData.append('ontology_file', ontologyFile);
      const result = await api.createKG(formData);
      dispatch({ type: 'SET_KG', kgId: result.kg_id, kgName: result.kg_name || null });
      if (result.kg_name) {
        safeSet('currentKGName', result.kg_name);
        setKgFilter(result.kg_name);
        dispatch({ type: 'SET_KG_LIST', kgList: [...state.kgList, { name: result.kg_name }] });
      }
      if (result.graph_data) dispatch({ type: 'SET_GRAPH_DATA', data: result.graph_data });
      setFile(null);
      if (fileRef.current) fileRef.current.value = '';
      setOntologyFile(null);
      if (ontologyRef.current) ontologyRef.current.value = '';
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
    const targetName = kgFilter || state.currentKGName;
    try {
      const result = await api.clearKG(targetName || undefined);
      const isLoadedKG = !targetName || targetName === state.currentKGName;
      if (isLoadedKG) {
        dispatch({ type: 'CLEAR_KG' });
        localStorage.removeItem('currentKGName');
        setKgName('');
      }
      if (targetName) {
        dispatch({ type: 'SET_KG_LIST', kgList: state.kgList.filter((kg) => kg.name !== targetName) });
      } else {
        dispatch({ type: 'SET_KG_LIST', kgList: [] });
      }
      setKgFilter('');
      showSuccess(dispatch, result.message || 'KG cleared');
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      showError(dispatch, `Failed to clear KG: ${msg}`);
    }
  };

  const handleLoadKG = () => {
    let loadMode: string;
    let nodeLimit: number;
    if (importMode === 'sample') {
      loadMode = 'sample';
      nodeLimit = 0;
    } else if (importMode === 'complete') {
      loadMode = 'complete';
      nodeLimit = 0;
    } else {
      loadMode = 'limited';
      nodeLimit = Number(importMode.split('-')[1]);
    }
    onLoadKG(loadMode, nodeLimit, kgFilter);
  };

  const refreshKGList = useCallback(() => {
    api.fetchKGList().then(
      (res) => dispatch({ type: 'SET_KG_LIST', kgList: res.kgs || [] }),
      () => {},
    );
  }, [dispatch]);

  return (
    <div className="flex flex-col gap-3 min-w-0">
      {/* ── Build Section ─────────────────────────────────── */}

      {/* File upload drop zone */}
      <input ref={fileRef} type="file" accept=".pdf,.txt,.json,.csv" className="hidden" onChange={handleFileChange} />
      <button
        type="button"
        className={[
          'flex w-full items-center gap-2 rounded-lg border px-3 py-2 text-left text-sm transition-colors',
          file
            ? 'border-file-selected-border text-base-content'
            : 'border-dashed border-base-content/30 text-base-content/50 hover:border-primary/50',
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
            ? 'border-file-selected-border text-base-content'
            : 'border-dashed border-base-content/30 text-base-content/50 hover:border-primary/50',
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
      <FieldsetDropdown
        label="Embedding Model"
        options={[
          { value: 'sentence_transformers', label: 'Sentence Transformers' },
          { value: 'openai', label: 'OpenAI Embeddings' },
          { value: 'vertexai', label: 'Google Vertex AI' },
          { value: 'titan', label: 'AWS Titan' },
        ]}
        value={embeddingModel}
        onChange={setEmbeddingModel}
      />

      {/* Max Chunks */}
      <fieldset className="border border-base-content/20 focus-within:border-primary/50 rounded-lg px-3 pb-2 pt-0 transition-colors">
        <legend className="text-2xs text-base-content/50 px-1 ml-auto mr-2">Max Chunks</legend>
        <input
          type="number"
          className="w-full bg-transparent text-sm outline-none"
          value={maxChunks}
          onChange={(e) => setMaxChunks(e.target.value)}
          min={1}
          max={100}
        />
      </fieldset>

      {/* KG Name */}
      <fieldset className="border border-base-content/20 focus-within:border-primary/50 rounded-lg px-3 pb-2 pt-0 transition-colors">
        <legend className="text-2xs text-base-content/50 px-1 ml-auto mr-2">KG Name</legend>
        <input
          type="text"
          className="w-full bg-transparent text-sm outline-none"
          placeholder={kgNamePlaceholder}
          value={kgName}
          onChange={(e) => setKgName(e.target.value)}
        />
      </fieldset>

      {/* Create KG */}
      <button
        type="button"
        className={[
          'btn btn-sm w-full shadow-none',
          creating || !canCreate
            ? 'bg-transparent border border-base-content/20 text-base-content/30 pointer-events-none'
            : 'bg-transparent border border-[color:oklch(62%_0.10_270)]/50 text-[color:oklch(62%_0.10_270)] hover:bg-[color:oklch(62%_0.10_270)] hover:text-white',
        ].join(' ')}
        onClick={handleCreate}
        disabled={creating || !canCreate}
      >
        {creating && <span className="loading loading-spinner loading-sm" />}
        {creating ? 'Creating KG...' : 'Create KG'}
      </button>

      {/* ── Decorative Separator ──────────────────────────── */}
      <div className="flex items-center gap-3 my-2">
        <div className="h-px flex-1 bg-gradient-to-r from-transparent to-base-content/20" />
        <div className="size-1 rounded-full bg-base-content/20" />
        <div className="h-px flex-1 bg-gradient-to-l from-transparent to-base-content/20" />
      </div>

      {/* ── Load Section ──────────────────────────────────── */}

      {/* Import Mode */}
      <FieldsetDropdown
        label="Import Mode"
        options={[
          { value: 'limited-500', label: 'First 500 nodes' },
          { value: 'limited-1000', label: 'First 1000 nodes' },
          { value: 'sample', label: 'Top hubs (smart sample)' },
          { value: 'complete', label: 'Complete graph' },
        ]}
        value={importMode}
        onChange={setImportMode}
      />

      {/* KG Name Filter */}
      <FieldsetDropdown
        label="KG Name"
        placeholder="Select KG"
        options={state.kgList.map((kg) => ({ value: kg.name, label: kg.name }))}
        value={kgFilter}
        onChange={setKgFilter}
        onOpen={refreshKGList}
      />

      {/* Load KG + Delete KG */}
      <div className="flex gap-2">
        <button
          type="button"
          className={[
            'btn btn-sm flex-1 shadow-none',
            kgFilter
              ? 'bg-transparent border border-[color:oklch(62%_0.10_270)]/50 text-[color:oklch(62%_0.10_270)] hover:bg-[color:oklch(62%_0.10_270)] hover:text-white'
              : 'bg-transparent border border-base-content/20 text-base-content/30 pointer-events-none',
          ].join(' ')}
          onClick={handleLoadKG}
          disabled={!kgFilter}
        >
          Load KG
        </button>
        <button
          type="button"
          className={[
            'btn btn-sm flex-1',
            kgFilter || state.currentKGId
              ? 'btn-outline border-error/50 text-error hover:bg-error hover:text-error-content'
              : 'bg-transparent border border-base-content/20 text-base-content/30 pointer-events-none',
          ].join(' ')}
          onClick={handleDelete}
          disabled={!kgFilter && !state.currentKGId}
        >
          Delete KG
        </button>
      </div>
    </div>
  );
}
