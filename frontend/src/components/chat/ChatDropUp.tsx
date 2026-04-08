import { useState, useRef, useEffect, useCallback } from 'react';
import { ChevronUpIcon } from '@heroicons/react/20/solid';

interface ChatDropUpProps {
  options: string[];
  value: string;
  onChange: (value: string) => void;
}

export function ChatDropUp({ options, value, onChange }: ChatDropUpProps) {
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

  const handleSelect = useCallback((option: string) => {
    onChange(option);
    setOpen(false);
  }, [onChange]);

  return (
    <div ref={containerRef} className="relative">
      {/* Options (above) */}
      {open && (
        <div className="absolute bottom-full left-0 mb-1 bg-base-100 border border-base-300 rounded-lg shadow-lg z-20 max-h-48 overflow-y-auto min-w-max">
          {options.map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => handleSelect(option)}
              className={`block w-full text-left px-3 py-1.5 text-xs hover:bg-base-200 transition-colors first:rounded-t-lg last:rounded-b-lg ${
                option === value ? 'text-primary font-semibold' : 'text-base-content'
              }`}
            >
              {option}
            </button>
          ))}
        </div>
      )}

      {/* Trigger button */}
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex items-center gap-0.5 px-2.5 py-1 rounded-full bg-base-300 text-xs text-base-content hover:bg-base-300/80 transition-colors whitespace-nowrap"
      >
        {value}
        <ChevronUpIcon className="size-3 opacity-50" />
      </button>
    </div>
  );
}
