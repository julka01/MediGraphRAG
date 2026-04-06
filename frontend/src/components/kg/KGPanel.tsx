import { useState } from 'react';
import { useApp } from '../../context/AppContext';
import { api } from '../../api';
import { FileUpload } from './FileUpload';
import { KGSelector } from './KGSelector';
import { showError, showSuccess } from '../ui/Notifications';

export function KGPanel({ kgModelHook, onNeo4jOpen, onProgressStart, onProgressStop }) {
  const { state, dispatch } = useApp();
  const [file, setFile] = useState(null);
  const [ontologyFile, setOntologyFile] = useState(null);
  const [selectedKG, setSelectedKG] = useState('');
  const [newKGName, setNewKGName] = useState('');
  const [embeddingModel, setEmbeddingModel] = useState('sentence_transformers');
  const [maxChunks, setMaxChunks] = useState(20);
  const [creating, setCreating] = useState(false);

  const handleCreate = async () => {
    if (!file) { showError(dispatch, 'Please select a file first'); return; }

    setCreating(true);
    onProgressStart();

    try {
      const kgName = selectedKG || newKGName;
      const formData = new FormData();
      formData.append('file', file);
      formData.append('provider', kgModelHook.vendor);
      formData.append('model', kgModelHook.selectedModel);
      formData.append('embedding_model', embeddingModel);
      formData.append('max_chunks', maxChunks);
      if (kgName) formData.append('kg_name', kgName);
      if (ontologyFile) formData.append('ontology_file', ontologyFile);

      const result = await api.createKG(formData);
      onProgressStop();

      dispatch({ type: 'SET_KG', kgId: result.kg_id, kgName: result.kg_name || null });
      if (result.kg_name) localStorage.setItem('currentKGName', result.kg_name);

      if (result.graph_data) dispatch({ type: 'SET_GRAPH_DATA', data: result.graph_data });

      setFile(null);
      showSuccess(dispatch, `KG created: ${result.graph_data?.nodes?.length || 0} nodes, ${result.graph_data?.relationships?.length || 0} edges`);
    } catch (error) {
      onProgressStop();
      showError(dispatch, `KG creation failed: ${error.message}`);
    } finally {
      setCreating(false);
    }
  };

  const handleClear = async () => {
    if (!confirm('⚠️ WARNING: This will permanently delete ALL nodes and relationships from Neo4j. Continue?')) return;

    try {
      const result = await api.clearKG();
      dispatch({ type: 'CLEAR_KG' });
      localStorage.removeItem('currentKGName');
      showSuccess(dispatch, result.message || 'KG cleared');
    } catch (error) {
      showError(dispatch, `Failed to clear KG: ${error.message}`);
    }
  };

  return (
    <div className="space-y-3">
      <FileUpload onFileSelected={setFile} onOntologySelected={setOntologyFile} />

      <div className="flex gap-2">
        <button className="btn btn-primary btn-sm flex-1" onClick={() => onNeo4jOpen()}>From Neo4j</button>
        <button className="btn btn-ghost btn-sm text-error" onClick={handleClear}>Clear KG</button>
      </div>

      {file && (
        <div className="space-y-2 border border-base-300 rounded-lg p-2">
          <KGSelector kgList={state.kgList} selectedKG={selectedKG} onSelectKG={(v) => { setSelectedKG(v); if (v) setNewKGName(''); }} />

          {selectedKG === '' && (
            <input type="text" className="input input-bordered input-sm w-full" placeholder="Enter new KG name" value={newKGName} onChange={(e) => setNewKGName(e.target.value)} />
          )}

          <fieldset className="fieldset">
            <legend className="fieldset-legend text-xs">Embedding Model</legend>
            <select className="select select-bordered select-sm w-full" value={embeddingModel} onChange={(e) => setEmbeddingModel(e.target.value)}>
              <option value="sentence_transformers">Sentence Transformers (Default)</option>
              <option value="openai">OpenAI Embeddings</option>
              <option value="vertexai">Google Vertex AI</option>
              <option value="titan">AWS Titan</option>
            </select>
          </fieldset>

          <fieldset className="fieldset">
            <legend className="fieldset-legend text-xs">Max Chunks per Report</legend>
            <input type="number" className="input input-bordered input-sm w-full" value={maxChunks} onChange={(e) => setMaxChunks(e.target.value)} min={1} max={100} />
          </fieldset>

          <button className="btn btn-primary btn-sm w-full" onClick={handleCreate} disabled={creating}>
            {creating ? <span className="loading loading-spinner loading-sm" /> : null}
            {creating ? 'Creating KG...' : 'Create KG'}
          </button>
        </div>
      )}
    </div>
  );
}
