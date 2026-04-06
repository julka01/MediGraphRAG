import { useState, useEffect } from 'react';
import { api } from '../../api';
import { useApp } from '../../context/AppContext';
import { showError } from '../ui/Notifications';

export function Neo4jForm({ open, onClose, onLoaded }) {
  const { state, dispatch } = useApp();
  const [uri, setUri] = useState('bolt://localhost:7687');
  const [user, setUser] = useState('neo4j');
  const [password, setPassword] = useState('');
  const [loadMode, setLoadMode] = useState('limited');
  const [nodeLimit, setNodeLimit] = useState(1000);
  const [kgFilter, setKgFilter] = useState('');
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    api.fetchDefaultCredentials()
      .then((data) => {
        if (data.uri) setUri(data.uri);
        if (data.user) setUser(data.user);
      })
      .catch(() => {});
  }, []);

  const handleConnect = async () => {
    if (!uri || !user) {
      showError(dispatch, 'Please fill in all required fields');
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('uri', uri);
      formData.append('user', user);
      formData.append('password', password);

      if (kgFilter) formData.append('kg_label', kgFilter);

      switch (loadMode) {
        case 'limited':
          formData.append('limit', nodeLimit);
          formData.append('sample_mode', 'false');
          formData.append('load_complete', 'false');
          break;
        case 'sample':
          formData.append('sample_mode', 'true');
          formData.append('load_complete', 'false');
          if (nodeLimit) formData.append('limit', nodeLimit);
          break;
        case 'complete':
          formData.append('load_complete', 'true');
          formData.append('sample_mode', 'false');
          break;
      }

      const result = await api.loadFromNeo4j(formData);
      const resultStats = result.stats || {};
      setStats(resultStats);

      onLoaded(result, kgFilter, resultStats);
      onClose();
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      showError(dispatch, `Neo4j loading failed: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  if (!open) return null;

  return (
    <dialog className="modal modal-open">
      <div className="modal-box max-w-md">
        <button className="btn btn-sm btn-circle btn-ghost absolute right-2 top-2" onClick={onClose}>✕</button>
        <h3 className="font-bold text-lg mb-4">Load from Neo4j</h3>

        <div className="space-y-3">
          <input type="text" className="input input-bordered input-sm w-full" placeholder="Neo4j URI" value={uri} onChange={(e) => setUri(e.target.value)} />
          <input type="text" className="input input-bordered input-sm w-full" placeholder="Username" value={user} onChange={(e) => setUser(e.target.value)} />
          <input type="password" className="input input-bordered input-sm w-full" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} />

          <fieldset className="fieldset">
            <legend className="fieldset-legend text-xs font-bold">Import Options</legend>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="radio" className="radio radio-sm" name="load-mode" value="limited" checked={loadMode === 'limited'} onChange={() => setLoadMode('limited')} />
              <span className="text-sm">Limited (1000 nodes max)</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="radio" className="radio radio-sm" name="load-mode" value="sample" checked={loadMode === 'sample'} onChange={() => setLoadMode('sample')} />
              <span className="text-sm">Smart Sample</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="radio" className="radio radio-sm" name="load-mode" value="complete" checked={loadMode === 'complete'} onChange={() => setLoadMode('complete')} />
              <span className="text-sm">Complete Import</span>
            </label>
          </fieldset>

          <fieldset className="fieldset">
            <legend className="fieldset-legend text-xs">Node Limit</legend>
            <input type="number" className="input input-bordered input-sm w-full" value={nodeLimit} onChange={(e) => setNodeLimit(e.target.value)} min={100} max={10000} step={100} />
          </fieldset>

          <fieldset className="fieldset">
            <legend className="fieldset-legend text-xs">Filter by KG Name</legend>
            <select className="select select-bordered select-sm w-full" value={kgFilter} onChange={(e) => setKgFilter(e.target.value)}>
              <option value="">All KGs</option>
              {state.kgList.map((kg) => (
                <option key={kg.name} value={kg.name}>{kg.name}</option>
              ))}
            </select>
          </fieldset>

          <button className="btn btn-success btn-sm w-full" onClick={handleConnect} disabled={loading}>
            {loading ? <span className="loading loading-spinner loading-sm" /> : null}
            {loading ? 'Loading...' : 'Load Knowledge Graph'}
          </button>

          {stats && (
            <div className="text-xs opacity-70 space-y-0.5">
              <div>Database: {stats.total_nodes_in_db || '-'} nodes, {stats.total_relationships_in_db || '-'} relationships</div>
              <div>Loaded: {stats.loaded_nodes || '-'} nodes, {stats.loaded_relationships || '-'} relationships</div>
            </div>
          )}
        </div>
      </div>
      <form method="dialog" className="modal-backdrop">
        <button onClick={onClose}>close</button>
      </form>
    </dialog>
  );
}
