import { useRef } from 'react';

interface ChatInputProps {
  onSend: (text: string) => void;
  disabled: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    const value = inputRef.current?.value?.trim();
    if (!value || disabled) return;
    onSend(value);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div className="flex gap-2">
      <textarea
        ref={inputRef}
        className="textarea textarea-bordered flex-1 text-sm min-h-10 max-h-32"
        placeholder="Ask a question…"
        onKeyDown={handleKeyDown}
        disabled={disabled}
        rows={1}
      />
      <button type="button" className="btn btn-primary btn-sm self-end" onClick={handleSend} disabled={disabled}>
        {disabled ? <span className="loading loading-spinner loading-xs" /> : 'Send'}
      </button>
    </div>
  );
}
