import { useState, useRef, useEffect, useCallback } from 'react';
import { ChevronUpIcon } from '@heroicons/react/20/solid';
import { shortenModelName } from '../../utils/models';

type Option = string | { value: string; label: string };

interface ChatDropUpProps {
  options: Option[];
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export function ChatDropUp({ options, value, onChange, placeholder }: ChatDropUpProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    function handleClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [open]);

  const optVal = (o: Option) => (typeof o === 'string' ? o : o.value);
  const optLabel = (o: Option) => (typeof o === 'string' ? shortenModelName(o) : o.label);

  const handleSelect = useCallback((option: Option) => {
    onChange(optVal(option));
    setOpen(false);
  }, [onChange]);

  const displayLabel = options.find((o) => optVal(o) === value);

  return (
    <div ref={containerRef} className="relative min-w-0">
      {/* Options (above) */}
      {open && (
        <div className="absolute bottom-full left-0 mb-1 bg-base-100 border border-base-300 rounded-lg shadow-lg z-20 max-h-48 overflow-y-auto min-w-max">
          {options.map((option) => (
            <button
              key={optVal(option)}
              type="button"
              onClick={() => handleSelect(option)}
              className={`block w-full text-left px-3 py-1.5 text-xs hover:bg-base-200 transition-colors first:rounded-t-lg last:rounded-b-lg ${
                optVal(option) === value ? 'text-[color:oklch(62%_0.10_270)] font-semibold' : 'text-base-content'
              }`}
            >
              {optLabel(option)}
            </button>
          ))}
        </div>
      )}

      {/* Trigger button */}
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex items-center gap-0.5 px-2.5 py-1 rounded-full bg-base-300 text-xs text-base-content hover:bg-base-300/80 transition-colors min-w-0"
      >
        <span className="truncate">{displayLabel ? optLabel(displayLabel) : (value ? shortenModelName(value) : (placeholder ?? '—'))}</span>
        <ChevronUpIcon className="size-3 shrink-0 opacity-50" />
      </button>
    </div>
  );
}
