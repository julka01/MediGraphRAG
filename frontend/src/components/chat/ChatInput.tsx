import { useCallback, useRef } from 'react';
import { ArrowUpIcon } from '@heroicons/react/20/solid';
import { ChatDropUp } from './ChatDropUp';
import type { UseModelsReturn } from '../../types/app';

const VENDORS = ['OpenAI', 'OpenRouter', 'LMU Lightllm', 'Ollama', 'Anthropic', 'Google'];

interface ChatInputProps {
  onSend: (text: string) => void;
  disabled: boolean;
  ragModelHook: UseModelsReturn;
}

export function ChatInput({ onSend, disabled, ragModelHook }: ChatInputProps) {
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(() => {
    const text = inputRef.current?.value.trim();
    if (!text || disabled) return;
    onSend(text);
    if (inputRef.current) {
      inputRef.current.value = '';
      inputRef.current.style.height = 'auto';
    }
  }, [onSend, disabled]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <div className="border border-base-300 rounded-xl p-2 has-focus-within:ring-1 has-focus-within:ring-primary/30">
      <textarea
        ref={inputRef}
        placeholder="Ask about the knowledge graph..."
        onKeyDown={handleKeyDown}
        disabled={disabled}
        className="w-full bg-transparent text-sm resize-none outline-none min-h-6 max-h-[33vh] overflow-y-auto field-sizing-content"
        rows={1}
      />
      <div data-chat-controls className="flex items-center justify-between mt-1.5">
        <div className="flex items-center gap-1.5">
          <ChatDropUp
            options={VENDORS}
            value={ragModelHook.vendor}
            onChange={(v) => ragModelHook.changeVendor(v)}
          />
          <ChatDropUp
            options={ragModelHook.models.length ? ragModelHook.models : [ragModelHook.selectedModel]}
            value={ragModelHook.selectedModel}
            onChange={(m) => ragModelHook.setSelectedModel(m)}
          />
        </div>
        <button
          type="button"
          onClick={handleSend}
          disabled={disabled}
          className="size-7 rounded-full bg-primary text-primary-content flex items-center justify-center hover:bg-primary/90 transition-colors disabled:opacity-40"
          aria-label="Send message"
        >
          {disabled ? (
            <span className="loading loading-spinner loading-xs" />
          ) : (
            <ArrowUpIcon className="size-4" />
          )}
        </button>
      </div>
    </div>
  );
}
