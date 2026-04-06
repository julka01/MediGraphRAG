import { useEffect } from 'react';
import { useHealth } from '../../hooks/useHealth';

export function HealthDot() {
  const { status, checking, checkHealth } = useHealth();

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  return (
    <div
      className="absolute top-3.5 right-3.5 w-2.5 h-2.5 rounded-full cursor-pointer transition-colors duration-300 z-10"
      style={{ background: checking ? 'var(--text-3, #888)' : status.color }}
      title={checking ? 'Rechecking…' : status.tip}
      onClick={checkHealth}
    />
  );
}
