import { XMarkIcon } from '@heroicons/react/24/outline';
import { useEffect, useState } from 'react';
import { api } from '../../api';
import { useApp } from '../../context/AppContext';
import { showError, showSuccess } from '../ui/Notifications';

const STORAGE_KEY = 'neo4j-credentials';

interface StoredCredentials {
  uri: string;
  user: string;
  password: string;
}

interface DatabaseSettingsProps {
  open: boolean;
  onClose: () => void;
}

export function DatabaseSettings({ open, onClose }: DatabaseSettingsProps) {
  const { dispatch } = useApp();
  const [uri, setUri] = useState('bolt://localhost:7687');
  const [user, setUser] = useState('neo4j');
  // Empty means no new password entered; placeholder shows ••• when saved creds exist
  const [password, setPassword] = useState('');
  const [hasSavedCreds, setHasSavedCreds] = useState(false);
  const [testing, setTesting] = useState(false);

  // On open: load saved credentials or fall back to server defaults
  useEffect(() => {
    if (!open) return;

    const stored = sessionStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const creds = JSON.parse(stored) as StoredCredentials;
        setUri(creds.uri);
        setUser(creds.user);
        setPassword(''); // show placeholder ••• instead of the real value
        setHasSavedCreds(true);
        return;
      } catch {
        // malformed — fall through to server defaults
      }
    }

    setHasSavedCreds(false);
    api
      .fetchDefaultCredentials()
      .then((data) => {
        if (data.uri) setUri(data.uri);
        if (data.user) setUser(data.user);
      })
      .catch(() => {});
  }, [open]);

  const resolvePassword = (): string => {
    // If the user typed something, use it; otherwise fall back to stored password
    if (password) return password;
    const stored = sessionStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        return (JSON.parse(stored) as StoredCredentials).password;
      } catch {
        return '';
      }
    }
    return '';
  };

  const handleTest = async () => {
    if (!uri || !user) {
      showError(dispatch, 'Please fill in URI and username');
      return;
    }
    const pw = resolvePassword();
    setTesting(true);
    try {
      const formData = new FormData();
      formData.append('uri', uri);
      formData.append('user', user);
      formData.append('password', pw);
      // Minimal load just to verify connectivity — limit to 1 node
      formData.append('limit', '1');
      formData.append('sample_mode', 'false');
      formData.append('load_complete', 'false');
      await api.loadFromNeo4j(formData);
      showSuccess(dispatch, 'Connection successful');
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      showError(dispatch, `Connection failed: ${msg}`);
    } finally {
      setTesting(false);
    }
  };

  // Credentials are stored in sessionStorage (cleared when browser closes).
  // For production, consider a backend-managed credential store.
  const handleSave = () => {
    if (!uri || !user) {
      showError(dispatch, 'Please fill in URI and username');
      return;
    }
    const pw = resolvePassword();
    const creds: StoredCredentials = { uri, user, password: pw };
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(creds));
    setHasSavedCreds(true);
    setPassword(''); // reset so placeholder shows again
    showSuccess(dispatch, 'Database settings saved');
    onClose();
  };

  if (!open) return null;

  return (
    <dialog className="modal modal-open">
      <div className="modal-box max-w-md">
        <button
          type="button"
          className="btn btn-sm btn-circle btn-ghost absolute right-2 top-2"
          onClick={onClose}
          aria-label="Close"
        >
          <XMarkIcon className="size-4" aria-hidden="true" />
        </button>
        <h3 className="font-bold text-lg mb-4">Database Settings</h3>
        <div className="space-y-3">
          <input
            type="text"
            className="input input-bordered input-sm w-full"
            placeholder="Neo4j URI"
            value={uri}
            onChange={(e) => setUri(e.target.value)}
          />
          <input
            type="text"
            className="input input-bordered input-sm w-full"
            placeholder="Username"
            value={user}
            onChange={(e) => setUser(e.target.value)}
          />
          <input
            type="password"
            className="input input-bordered input-sm w-full"
            placeholder={hasSavedCreds ? '•••••••••••' : 'Password'}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <div className="flex gap-2 pt-1">
            <button
              type="button"
              className="btn btn-outline btn-sm flex-1"
              onClick={handleTest}
              disabled={testing}
            >
              {testing ? <span className="loading loading-spinner loading-xs" /> : null}
              {testing ? 'Testing…' : 'Test'}
            </button>
            <button
              type="button"
              className="btn btn-primary btn-sm flex-1"
              onClick={handleSave}
            >
              Save
            </button>
          </div>
        </div>
      </div>
      <form method="dialog" className="modal-backdrop">
        <button type="submit" onClick={onClose}>
          close
        </button>
      </form>
    </dialog>
  );
}
