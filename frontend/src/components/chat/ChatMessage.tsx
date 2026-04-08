import clsx from 'clsx';
import DOMPurify from 'dompurify';
import { memo, useState } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ChatMessageProps {
  message: string;
  type: string;
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

  return (
    <div className={clsx('chat', isUser ? 'chat-end' : 'chat-start')}>
      <div
        className={clsx('chat-bubble rounded-2xl text-sm', {
          'chat-bubble-primary': isUser,
          'chat-bubble-error': isError,
          'chat-bubble-ghost': isThinking,
        })}
      >
        {isThinking && <span className="loading loading-dots loading-xs" />}
        {isUser || isError ? (
          <span>{message}</span>
        ) : !isThinking ? (
          <Markdown remarkPlugins={[remarkGfm]}>{message}</Markdown>
        ) : null}
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
