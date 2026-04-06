import { useState } from 'react';

interface ChatMessageProps {
  message: string;
  type: string;
  timestamp?: string;
}

export function ChatMessage({ message, type, timestamp }: ChatMessageProps) {
  const [copied, setCopied] = useState(false);
  const isUser = type === 'user';
  const isAI = type === 'ai';
  const isThinking = type === 'thinking';
  const isError = type === 'error';

  const handleCopy = () => {
    const tmp = document.createElement('div');
    tmp.innerHTML = message;
    navigator.clipboard.writeText(tmp.textContent || '').then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <div className={`chat ${isUser ? 'chat-end' : 'chat-start'}`}>
      <div className={`chat-bubble ${isUser ? 'chat-bubble-primary' : isError ? 'chat-bubble-error' : isThinking ? 'chat-bubble-ghost' : ''} text-sm`}>
        {isThinking && <span className="loading loading-dots loading-xs" />}
        {(isUser || isError) ? <span>{message}</span> : !isThinking ? <div dangerouslySetInnerHTML={{ __html: message }} /> : null}
        {isAI && <button className="btn btn-ghost btn-xs opacity-50 hover:opacity-100 mt-1" onClick={handleCopy}>{copied ? 'copied!' : 'copy'}</button>}
      </div>
      {timestamp && <div className="chat-footer opacity-50 text-xs">{timestamp}</div>}
    </div>
  );
}
