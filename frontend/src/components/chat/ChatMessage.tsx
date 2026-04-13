import clsx from 'clsx';
import DOMPurify from 'dompurify';
import { memo, useState } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ChatMessageProps {
  message: string;
  type: 'user' | 'ai' | 'thinking' | 'error';
  timestamp?: string;
}

export const ChatMessage = memo(function ChatMessage({ message, type, timestamp }: ChatMessageProps) {
  const [copied, setCopied] = useState(false);
  const isUser = type === 'user';
  const isAI = type === 'ai';
  const isThinking = type === 'thinking';
  const isError = type === 'error';

  const handleCopy = () => {
    const tmp = document.createElement('div');
    tmp.innerHTML = DOMPurify.sanitize(message);
    navigator.clipboard.writeText(tmp.textContent || '').then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  if (isThinking) {
    return (
      <div className="chat chat-start">
        <div className="rounded-2xl bg-base-300 px-4 py-2 text-sm max-w-[80%]">
          <span
            className="loading loading-dots loading-xs text-base-content"
            aria-label="AI is thinking"
            role="status"
          />
        </div>
      </div>
    );
  }

  return (
    <div className={clsx('chat', isUser ? 'chat-end' : 'chat-start')}>
      <div
        className={clsx('chat-bubble before:hidden rounded-2xl text-sm break-words', {
          'bg-[color:oklch(62%_0.10_270)]/50 text-base-content': isUser,
          'chat-bubble-error': isError,
        })}
      >
        {isUser || isError ? <span>{message}</span> : <Markdown remarkPlugins={[remarkGfm]}>{message}</Markdown>}
        {isAI && (
          <button type="button" className="btn btn-ghost btn-xs opacity-50 hover:opacity-100 mt-1" onClick={handleCopy}>
            {copied ? 'copied!' : 'copy'}
          </button>
        )}
      </div>
      {timestamp && <div className="chat-footer opacity-50 text-xs">{timestamp}</div>}
    </div>
  );
});
