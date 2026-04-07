import { useEffect, useRef } from 'react';
import { useHealth } from '../../hooks/useHealth';
import type { HealthResponse } from '../../types/app';

interface HealthDotProps {
  initialData?: HealthResponse | null;
}

const statusColorClass: Record<string, string> = {
  ok: 'status-success',
  warn: 'status-warning',
  fail: 'status-error',
};

export function HealthDot({ initialData }: HealthDotProps) {
  const { status, checking, checkHealth } = useHealth(initialData);
  const hasInitialData = useRef(!!initialData);

  useEffect(() => {
    if (!hasInitialData.current) {
      checkHealth();
    }
  }, [checkHealth]);

  const colorClass = checking ? 'status-info' : (statusColorClass[status.level] ?? 'status-info');

  return (
    <button
      type="button"
      className={`absolute top-3.5 right-3.5 status status-sm ${colorClass} cursor-pointer z-10 border-0 p-0`}
      title={checking ? 'Rechecking…' : status.tip}
      onClick={checkHealth}
      aria-label={checking ? 'Rechecking health…' : status.tip}
    />
  );
}
