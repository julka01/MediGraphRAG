import { useState, useEffect, useRef } from 'react';

interface LogEntry {
  text: string;
  cls: string;
}

interface ProgressPanelProps {
  active: boolean;
  onClose: () => void;
}

export function ProgressPanel({ active, onClose }: ProgressPanelProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const logRef = useRef<HTMLDivElement>(null);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!active) {
      if (sourceRef.current) {
        sourceRef.current.close();
        sourceRef.current = null;
      }
      return;
    }
    setLogs([]);
    const source = new EventSource('/kg_progress_stream');
    sourceRef.current = source;
    source.onmessage = (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data as string) as { done?: boolean; line?: string };
        if (data.done) {
          setLogs((prev) => [...prev, { text: '✓ Done', cls: 'text-success' }]);
          source.close();
          sourceRef.current = null;
          return;
        }
        if (data.line) {
          let cls = '';
          if (data.line.startsWith('❌') || data.line.startsWith('Error')) cls = 'text-error';
          else if (data.line.startsWith('✓') || data.line.startsWith('🎉')) cls = 'text-success';
          else if (data.line.startsWith('🔍') || data.line.startsWith('📊')) cls = 'text-info';
          setLogs((prev) => [...prev, { text: data.line!, cls }]);
        }
      } catch { /* ignore */ }
    };
    source.onerror = () => {
      source.close();
      sourceRef.current = null;
    };
    return () => {
      source.close();
      sourceRef.current = null;
    };
  }, [active]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  if (!active && logs.length === 0) return null;

  return (
    <div className="absolute inset-x-2 bottom-2 bg-base-200 border border-base-300 rounded-lg shadow-lg z-20 max-h-48 flex flex-col">
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-base-300">
        <span className="text-xs font-semibold">⚙ Building knowledge graph…</span>
        <button className="btn btn-ghost btn-xs" onClick={onClose}>✕</button>
      </div>
      <div ref={logRef} className="overflow-y-auto flex-1 p-2 text-xs font-mono space-y-0.5">
        {logs.map((log, i) => (
          <p key={i} className={log.cls}>{log.text}</p>
        ))}
      </div>
    </div>
  );
}
