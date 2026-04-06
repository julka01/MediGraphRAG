import { useRef } from 'react';

export function ChatInput({ onSend, disabled }) {
  const inputRef = useRef(null);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    const value = inputRef.current?.value?.trim();
    if (!value || disabled) return;
    onSend(value);
    inputRef.current.value = '';
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
      <button className="btn btn-primary btn-sm self-end" onClick={handleSend} disabled={disabled}>
        {disabled ? <span className="loading loading-spinner loading-xs" /> : 'Send'}
      </button>
    </div>
  );
}
