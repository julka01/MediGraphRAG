import { useState, useRef, useCallback } from 'react';
import { api } from '../api';

const HISTORY_KEY = 'kg-chat-history';
const MAX_HISTORY = 60;

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); }
  catch { return []; }
}

function saveHistory(messages) {
  try {
    const toSave = messages.filter((m) => m.type === 'user' || m.type === 'ai').slice(-MAX_HISTORY);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(toSave));
  } catch { /* quota exceeded */ }
}

export function useChat() {
  const [messages, setMessages] = useState(() => loadHistory());
  const [sending, setSending] = useState(false);
  const abortRef = useRef(null);

  const addMessage = useCallback((msg) => {
    setMessages((prev) => {
      const next = [...prev, msg];
      saveHistory(next);
      return next;
    });
  }, []);

  const sendQuestion = useCallback(async (question, kgName, vendor, model) => {
    addMessage({ type: 'user', message: `You: ${question}`, ts: Date.now() });
    setSending(true);

    try {
      const payload = { question, provider_rag: vendor, model_rag: model };
      if (kgName) payload.kg_name = kgName;

      abortRef.current = new AbortController();
      const chatTimeout = setTimeout(() => abortRef.current.abort(), 130000);

      let result;
      try {
        result = await api.sendChat(payload, abortRef.current.signal);
      } finally {
        clearTimeout(chatTimeout);
      }

      return result;
    } finally {
      setSending(false);
    }
  }, [addMessage]);

  const clearChat = useCallback(() => {
    setMessages([]);
    localStorage.removeItem(HISTORY_KEY);
  }, []);

  const exportChat = useCallback((kgName) => {
    if (messages.length === 0) return;
    let md = `# Chat Export — ${kgName || 'Knowledge Graph'}\n_Exported ${new Date().toLocaleString()}_\n\n`;
    messages.forEach((m) => {
      if (m.type === 'user') {
        md += `**You:** ${m.message.replace(/^You: /, '')}\n\n`;
      } else if (m.type === 'ai') {
        const tmp = document.createElement('div');
        tmp.innerHTML = m.message;
        md += `**Assistant:** ${tmp.textContent}\n\n---\n\n`;
      }
    });
    const blob = new Blob([md], { type: 'text/markdown' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `chat_${(kgName || 'export').replace(/\s+/g, '_')}_${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, [messages]);

  return { messages, sending, addMessage, sendQuestion, clearChat, exportChat };
}
