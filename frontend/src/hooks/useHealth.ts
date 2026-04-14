import { useCallback, useState } from 'react';
import { api } from '../api';
import type { HealthResponse, UseHealthReturn } from '../types/app';

function healthToStatus(d: HealthResponse): { level: string; tip: string } {
  const failed = (d.checks || [])
    .filter((c) => c.status === 'fail')
    .map((c) => c.check)
    .join(', ');
  const warned = (d.checks || [])
    .filter((c) => c.status === 'warn')
    .map((c) => c.check)
    .join(', ');
  let tip = `System: ${d.status.toUpperCase()}`;
  if (failed) tip += `\nFailed: ${failed}`;
  if (warned) tip += `\nWarnings: ${warned}`;
  return { level: d.status, tip };
}

export function useHealth(initialData?: HealthResponse | null): UseHealthReturn {
  const [status, setStatus] = useState<{ level: string; tip: string }>(() =>
    initialData ? healthToStatus(initialData) : { level: 'unknown', tip: 'Checking system health…' },
  );
  const [checking, setChecking] = useState(false);

  const checkHealth = useCallback(async () => {
    setChecking(true);
    try {
      const d = await api.checkHealth();
      setStatus(healthToStatus(d));
    } catch {
      setStatus({ level: 'fail', tip: 'Health check failed — server may be unreachable' });
    } finally {
      setChecking(false);
    }
  }, []);

  return { status, checking, checkHealth };
}
