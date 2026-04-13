import { ChevronDownIcon } from '@heroicons/react/20/solid';
import { useCallback, useEffect, useRef, useState } from 'react';

interface Option {
  value: string;
  label: string;
}

interface FieldsetDropdownProps {
  label: string;
  options: Option[];
  value: string;
  onChange: (value: string) => void;
  onOpen?: () => void;
  placeholder?: string;
  disabled?: boolean;
}

export function FieldsetDropdown({
  label,
  options,
  value,
  onChange,
  onOpen,
  placeholder,
  disabled,
}: FieldsetDropdownProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLFieldSetElement>(null);

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

  const handleSelect = useCallback(
    (val: string) => {
      onChange(val);
      setOpen(false);
    },
    [onChange],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (disabled) return;

      const currentIndex = options.findIndex((o) => o.value === value);

      if (!open) {
        if (['ArrowDown', 'ArrowUp', 'Enter', ' '].includes(e.key)) {
          e.preventDefault();
          onOpen?.();
          setOpen(true);
        }
        return;
      }

      switch (e.key) {
        case 'ArrowDown': {
          e.preventDefault();
          const nextIndex = currentIndex < options.length - 1 ? currentIndex + 1 : 0;
          onChange(options[nextIndex].value);
          break;
        }
        case 'ArrowUp': {
          e.preventDefault();
          const prevIndex = currentIndex > 0 ? currentIndex - 1 : options.length - 1;
          onChange(options[prevIndex].value);
          break;
        }
        case 'Escape':
          e.preventDefault();
          setOpen(false);
          break;
        case 'Enter':
        case ' ':
          e.preventDefault();
          setOpen(false);
          break;
      }
    },
    [disabled, open, options, value, onChange, onOpen],
  );

  const selected = options.find((o) => o.value === value);
  const displayLabel = selected?.label ?? (value || placeholder || '—');

  return (
    <fieldset
      ref={containerRef}
      onKeyDown={handleKeyDown}
      className={`fieldset relative border rounded-lg px-3 pb-2 pt-0 transition-colors min-w-0 ${open ? 'border-primary/50' : 'border-base-content/20'}`}
    >
      <legend className="text-2xs text-base-content/50 px-1 ml-auto mr-2">{label}</legend>
      <button
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => {
          if (disabled) return;
          if (!open) onOpen?.();
          setOpen(!open);
        }}
        disabled={disabled}
        className="flex w-full items-center justify-between text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 rounded text-left disabled:opacity-50"
      >
        <span className="truncate">{displayLabel}</span>
        <ChevronDownIcon
          aria-hidden="true"
          className={`size-4 shrink-0 text-base-content/40 transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>
      {open && (
        <div
          role="listbox"
          aria-label={label}
          className="absolute left-0 right-0 top-full mt-1 bg-base-100 border border-base-300 rounded-lg shadow-lg z-30 max-h-72 overflow-y-auto"
        >
          {options.length > 0 ? (
            options.map((option) => (
              <button
                key={option.value}
                type="button"
                role="option"
                aria-selected={option.value === value}
                onClick={() => handleSelect(option.value)}
                className={`block w-full text-left px-3 py-1.5 text-sm hover:bg-base-200 transition-colors first:rounded-t-lg last:rounded-b-lg ${
                  option.value === value ? 'text-primary font-semibold' : 'text-base-content'
                }`}
              >
                {option.label}
              </button>
            ))
          ) : (
            <span className="block px-3 py-1.5 text-sm text-base-content/50">No KGs available</span>
          )}
        </div>
      )}
    </fieldset>
  );
}
