import { useState, useCallback } from 'react';
import { api } from '../api';
import type { UseHealthReturn } from '../types/app';

export function useHealth(): UseHealthReturn {
  const [status, setStatus] = useState<{ color: string; tip: string }>({ color: '#888', tip: 'Checking system health…' });
  const [checking, setChecking] = useState(false);

  const checkHealth = useCallback(async () => {
    setChecking(true);
    try {
      const d = await api.checkHealth();
      const colors: Record<string, string> = { ok: '#2ecc71', warn: '#f39c12', fail: '#e74c3c' };
      const color = colors[d.status] || '#888';
      const failed = (d.checks || []).filter(c => c.status === 'fail').map(c => c.check).join(', ');
      const warned = (d.checks || []).filter(c => c.status === 'warn').map(c => c.check).join(', ');
      let tip = `System: ${d.status.toUpperCase()}`;
      if (failed) tip += `\nFailed: ${failed}`;
      if (warned) tip += `\nWarnings: ${warned}`;
      setStatus({ color, tip });
    } catch {
      setStatus({ color: '#e74c3c', tip: 'Health check failed — server may be unreachable' });
    } finally {
      setChecking(false);
    }
  }, []);

  return { status, checking, checkHealth };
}
