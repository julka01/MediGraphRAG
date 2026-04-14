import { useCallback, useRef, useEffect } from 'react';
import { ArrowUpIcon } from '@heroicons/react/20/solid';
import { ChatDropUp } from './ChatDropUp';
import type { UseModelsReturn } from '../../types/app';

const VENDORS = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'openrouter', label: 'OpenRouter' },
  { value: 'lmu_lightllm', label: 'LMU Lightllm' },
  { value: 'ollama', label: 'Ollama' },
  { value: 'anthropic', label: 'Anthropic' },
  { value: 'google', label: 'Google' },
];

interface ChatInputProps {
  onSend: (text: string) => boolean;
  disabled: boolean;
  ragModelHook: UseModelsReturn;
}

export function ChatInput({ onSend, disabled, ragModelHook }: ChatInputProps) {
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(() => {
    const text = inputRef.current?.value.trim();
    if (!text || disabled) return;
    const accepted = onSend(text);
    if (accepted && inputRef.current) {
      inputRef.current.value = '';
      inputRef.current.style.height = 'auto';
    }
  }, [onSend, disabled]);

  const autoResize = useCallback(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const maxH = window.innerHeight / 3;
    el.style.height = `${Math.min(el.scrollHeight, maxH)}px`;
  }, []);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  const handleInput = useCallback(() => {
    autoResize();
  }, [autoResize]);

  // Resize on mount in case of pre-filled content
  useEffect(() => {
    autoResize();
  }, [autoResize]);

  return (
    <div className="rounded-2xl border border-base-content/12 bg-base-100/70 p-2.5 shadow-sm transition-all focus-within:border-primary/50 focus-within:shadow-lg focus-within:shadow-primary/8">
      <textarea
        ref={inputRef}
        aria-label="Ask about the knowledge graph"
        name="chat-question"
        autoComplete="off"
        placeholder="Ask about the knowledge graph"
        onKeyDown={handleKeyDown}
        onInput={handleInput}
        disabled={disabled}
        className="min-h-6 max-h-[33vh] w-full resize-none overflow-y-auto bg-transparent text-sm leading-6 focus-visible:outline-none"
        rows={1}
      />
      <div data-chat-controls className="mt-2 flex items-center gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <ChatDropUp options={VENDORS} value={ragModelHook.vendor} onChange={(v) => ragModelHook.changeVendor(v)} />
          <ChatDropUp
            options={ragModelHook.models}
            value={ragModelHook.selectedModel}
            onChange={(m) => ragModelHook.setSelectedModel(m)}
            placeholder={ragModelHook.loading ? 'Loading…' : 'No models'}
          />
        </div>
        <button
          type="button"
          onClick={handleSend}
          disabled={disabled}
          className="ml-auto flex size-9 shrink-0 items-center justify-center rounded-full bg-primary text-primary-content shadow-lg shadow-primary/15 transition-all hover:-translate-y-px hover:brightness-105 disabled:opacity-40"
          aria-label="Send message"
        >
          {disabled ? <span className="loading loading-spinner loading-xs" /> : <ArrowUpIcon className="size-4" />}
        </button>
      </div>
    </div>
  );
}
