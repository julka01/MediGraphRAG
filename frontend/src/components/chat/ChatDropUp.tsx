import { ChevronUpIcon } from '@heroicons/react/20/solid';
import { useCallback, useEffect, useRef, useState } from 'react';
import { shortenModelName } from '../../utils/models';

type Option = string | { value: string; label: string };

interface ChatDropUpProps {
  options: Option[];
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

const optVal = (o: Option) => (typeof o === 'string' ? o : o.value);
const optLabel = (o: Option) => (typeof o === 'string' ? shortenModelName(o) : o.label);

export function ChatDropUp({ options, value, onChange, placeholder }: ChatDropUpProps) {
  const [open, setOpen] = useState(false);
  const [focusedIndex, setFocusedIndex] = useState(-1);
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

  // Reset focused index when opening, default to current selection
  useEffect(() => {
    if (open) {
      const idx = options.findIndex((o) => optVal(o) === value);
      setFocusedIndex(idx === -1 ? 0 : idx);
    }
  }, [open, options, value]);

  const handleSelect = useCallback(
    (option: Option) => {
      onChange(optVal(option));
      setOpen(false);
    },
    [onChange],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (!open) {
        if (['ArrowDown', 'ArrowUp', 'Enter', ' '].includes(e.key)) {
          e.preventDefault();
          setOpen(true);
        }
        return;
      }

      switch (e.key) {
        case 'ArrowDown': {
          e.preventDefault();
          setFocusedIndex((prev) => (prev + 1) % options.length);
          break;
        }
        case 'ArrowUp': {
          e.preventDefault();
          setFocusedIndex((prev) => (prev - 1 + options.length) % options.length);
          break;
        }
        case 'Enter':
        case ' ': {
          e.preventDefault();
          if (focusedIndex >= 0 && focusedIndex < options.length) {
            handleSelect(options[focusedIndex]);
          }
          break;
        }
        case 'Escape': {
          e.preventDefault();
          setOpen(false);
          break;
        }
      }
    },
    [open, options, focusedIndex, handleSelect],
  );

  const displayLabel = options.find((o) => optVal(o) === value);

  return (
    <div ref={containerRef} className="relative min-w-0">
      {/* Options (above) */}
      {open && (
        <div
          role="listbox"
          className="absolute bottom-full left-0 mb-1 bg-base-100 border border-base-300 rounded-lg shadow-lg z-20 max-h-48 overflow-y-auto min-w-max"
        >
          {options.map((option, index) => (
            <button
              key={optVal(option)}
              type="button"
              role="option"
              aria-selected={optVal(option) === value}
              onClick={() => handleSelect(option)}
              className={`block w-full text-left px-3 py-1.5 text-xs hover:bg-base-200 transition-colors first:rounded-t-lg last:rounded-b-lg ${
                optVal(option) === value ? 'text-brand font-semibold' : 'text-base-content'
              } ${index === focusedIndex ? 'bg-base-200' : ''}`}
            >
              {optLabel(option)}
            </button>
          ))}
        </div>
      )}

      {/* Trigger button */}
      <button
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        onKeyDown={handleKeyDown}
        className="flex items-center gap-0.5 px-2.5 py-1 rounded-full bg-base-300 text-xs text-base-content hover:bg-base-300/80 transition-colors min-w-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50"
      >
        <span className="truncate">
          {displayLabel ? optLabel(displayLabel) : value ? shortenModelName(value) : (placeholder ?? '—')}
        </span>
        <ChevronUpIcon aria-hidden="true" className="size-3 shrink-0 opacity-50" />
      </button>
    </div>
  );
}
