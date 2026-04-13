// frontend/src/components/kg/KGBuildSection.tsx
import clsx from 'clsx';
import { ArrowUpTrayIcon, CheckIcon } from '@heroicons/react/24/outline';
import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../../api';
import { useApp } from '../../context/AppContext';
import type { KGSettings, UseModelsReturn } from '../../types/app';
import { safeSet } from '../../utils/storage';
import { FieldsetDropdown } from '../ui/FieldsetDropdown';
import { showError, showSuccess } from '../ui/Notifications';
import { ModelSelector } from './ModelSelector';

const EMBEDDING_OPTIONS = ['sentence_transformers', 'openai', 'vertexai', 'titan'];

interface KGBuildSectionProps {
  kgModelHook: UseModelsReturn;
  onProgressStart: () => void;
  onProgressStop: () => void;
  loadedKGSettings?: KGSettings | null;
}

export function KGBuildSection({
  kgModelHook,
  onProgressStart,
  onProgressStop,
  loadedKGSettings,
}: KGBuildSectionProps) {
  const { state, dispatch } = useApp();
  const fileRef = useRef<HTMLInputElement>(null);
  const ontologyRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [ontologyFile, setOntologyFile] = useState<File | null>(null);
  const [kgName, setKgName] = useState('');
  const [embeddingModel, setEmbeddingModel] = useState('sentence_transformers');
  const [maxChunks, setMaxChunks] = useState('20');
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    if (!loadedKGSettings) return;
    if (loadedKGSettings.embeddingModel && EMBEDDING_OPTIONS.includes(loadedKGSettings.embeddingModel)) {
      setEmbeddingModel(loadedKGSettings.embeddingModel);
    }
    if (loadedKGSettings.maxChunks != null) {
      setMaxChunks(String(loadedKGSettings.maxChunks));
    }
  }, [loadedKGSettings]);

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

  const handleCreate = useCallback(async () => {
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
  }, [
    file,
    ontologyFile,
    kgName,
    embeddingModel,
    maxChunks,
    kgModelHook.vendor,
    kgModelHook.selectedModel,
    state.kgList,
    dispatch,
    onProgressStart,
    onProgressStop,
  ]);

  return (
    <>
      {/* File upload drop zone */}
      <input
        ref={fileRef}
        type="file"
        accept=".pdf,.txt,.json,.csv"
        className="hidden"
        onChange={handleFileChange}
        aria-label="Select document file"
      />
      <button
        type="button"
        className={clsx(
          'flex w-full items-center gap-2 rounded-lg border px-3 py-2 text-left text-sm transition-colors',
          file
            ? 'border-file-selected-border text-base-content'
            : 'border-dashed border-base-content/30 text-base-content/50 hover:border-primary/50',
        )}
        onClick={() => fileRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleFileDrop}
      >
        {file ? (
          <CheckIcon className="size-4 shrink-0 text-success" aria-hidden="true" />
        ) : (
          <ArrowUpTrayIcon className="size-4 shrink-0" aria-hidden="true" />
        )}
        <span className="truncate">{file ? `File: ${file.name}` : 'Drop or select file'}</span>
      </button>

      {/* Ontology upload drop zone */}
      <input
        ref={ontologyRef}
        type="file"
        accept=".json,.owl"
        className="hidden"
        onChange={handleOntologyChange}
        aria-label="Select ontology file"
      />
      <button
        type="button"
        className={clsx(
          'flex w-full items-center gap-2 rounded-lg border px-3 py-2 text-left text-sm transition-colors',
          ontologyFile
            ? 'border-file-selected-border text-base-content'
            : 'border-dashed border-base-content/30 text-base-content/50 hover:border-primary/50',
        )}
        onClick={() => ontologyRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleOntologyDrop}
      >
        {ontologyFile ? (
          <CheckIcon className="size-4 shrink-0 text-success" aria-hidden="true" />
        ) : (
          <ArrowUpTrayIcon className="size-4 shrink-0" aria-hidden="true" />
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
      <fieldset className="fieldset border rounded-lg px-3 pb-2 pt-0 border-base-content/20 focus-within:border-primary/50 transition-colors">
        <legend className="fieldset-legend text-2xs text-base-content/50 px-1 ml-auto mr-2">Max Chunks</legend>
        <input
          type="number"
          name="max-chunks"
          autoComplete="off"
          className="w-full bg-transparent text-sm focus-visible:outline-none"
          value={maxChunks}
          onChange={(e) => setMaxChunks(e.target.value)}
          min={1}
          max={100}
        />
      </fieldset>

      {/* KG Name */}
      <fieldset className="fieldset border rounded-lg px-3 pb-2 pt-0 border-base-content/20 focus-within:border-primary/50 transition-colors">
        <legend className="fieldset-legend text-2xs text-base-content/50 px-1 ml-auto mr-2">KG Name</legend>
        <input
          type="text"
          name="kg-name"
          autoComplete="off"
          className="w-full bg-transparent text-sm focus-visible:outline-none"
          placeholder={`${kgNamePlaceholder}\u2026`}
          value={kgName}
          onChange={(e) => setKgName(e.target.value)}
        />
      </fieldset>

      {/* Create KG */}
      <button
        type="button"
        className={clsx(
          'btn btn-sm w-full shadow-none',
          creating || !canCreate
            ? 'bg-transparent border border-base-content/20 text-base-content/30 pointer-events-none'
            : 'bg-transparent border border-primary/50 text-primary hover:bg-primary hover:text-primary-content',
        )}
        onClick={handleCreate}
        disabled={creating || !canCreate}
      >
        {creating && <span className="loading loading-spinner loading-sm" />}
        {creating ? 'Creating KG\u2026' : 'Create KG'}
      </button>
    </>
  );
}
