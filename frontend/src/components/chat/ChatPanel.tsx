import { useCallback, useEffect, useRef } from 'react';
import { useApp } from '../../context/AppContext';
import { useChat } from '../../hooks/useChat';
import type { ResponseSections as ResponseSectionsType, UseModelsReturn } from '../../types/app';
import { ChatInput } from './ChatInput';
import { ChatMessage } from './ChatMessage';
import { ChatSuggestions } from './ChatSuggestions';
import { ResponseSections, SourcesSection } from './ResponseSections';

function parseResponse(text: string): ResponseSectionsType {
  const sections: ResponseSectionsType = {
    recommendation: '',
    reasoning: '',
    evidence: '',
    nextSteps: '',
    fallback: '',
  };
  const lines = text.split('\n');
  let currentSection: keyof ResponseSectionsType | '' = '';
  let currentContent: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.toUpperCase().includes('RECOMMENDATION/SUMMARY') && trimmed.includes('#')) {
      if (currentSection) sections[currentSection] = currentContent.join('\n').trim();
      currentContent = [];
      currentSection = 'recommendation';
    } else if (trimmed.toUpperCase().includes('REASONING PATH') && trimmed.includes('#')) {
      if (currentSection) sections[currentSection] = currentContent.join('\n').trim();
      currentContent = [];
      currentSection = 'reasoning';
    } else if (trimmed.toUpperCase().includes('COMBINED EVIDENCE') && trimmed.includes('#')) {
      if (currentSection) sections[currentSection] = currentContent.join('\n').trim();
      currentContent = [];
      currentSection = 'evidence';
    } else if (trimmed.toUpperCase().includes('NEXT STEPS') && trimmed.includes('#')) {
      if (currentSection) sections[currentSection] = currentContent.join('\n').trim();
      currentContent = [];
      currentSection = 'nextSteps';
    } else if (currentSection && trimmed) {
      currentContent.push(trimmed);
    }
  }
  if (currentSection) sections[currentSection] = currentContent.join('\n').trim();
  if (!sections.recommendation && !sections.evidence && !sections.reasoning && !sections.nextSteps) {
    sections.fallback = text;
  }
  return sections;
}

interface ChatPanelProps {
  ragModelHook: UseModelsReturn;
}

export function ChatPanel({ ragModelHook }: ChatPanelProps) {
  const { state, dispatch } = useApp();
  const { messages, sending, addMessage, sendQuestion, clearChat, exportChat } = useChat();
  const chatBoxRef = useRef<HTMLDivElement>(null);

  // biome-ignore lint/correctness/useExhaustiveDependencies: messages triggers auto-scroll on new messages
  useEffect(() => {
    if (chatBoxRef.current) {
      const box = chatBoxRef.current;
      const dist = box.scrollHeight - box.scrollTop - box.clientHeight;
      if (dist < 80) box.scrollTop = box.scrollHeight;
    }
  }, [messages]);

  const handleClearChat = useCallback(() => clearChat(), [clearChat]);
  const handleExportChat = useCallback(() => exportChat(state.currentKGName), [exportChat, state.currentKGName]);

  const handleSend = useCallback(
    async (question: string) => {
      try {
        const result = await sendQuestion(
          question,
          state.currentKGName,
          ragModelHook.vendor,
          ragModelHook.selectedModel,
        );

        const usedEntities = result.info?.entities?.used_entities || [];
        const nodeNames = new Set<string>();
        usedEntities.forEach((entity) => {
          const readable = (entity.description || '').toLowerCase().trim();
          const idKey = (entity.id || '').toLowerCase().trim();
          if (readable) nodeNames.add(readable);
          if (idKey) nodeNames.add(idKey);
        });
        dispatch({ type: 'SET_HIGHLIGHTED_NODES', nodes: nodeNames });

        const responseText = result.response || result.message || 'No response generated';
        const cleanedText = responseText.replace(/【Source:[^】]*】/g, '').replace(/\[Source:[^\]]*\]/g, '');
        const sections = parseResponse(cleanedText);

        const confidence = result.info?.confidence_score;
        const entityCount = usedEntities.length;
        let sourceChip = '';
        if (entityCount > 0) {
          const pct = confidence !== undefined ? `${Math.round(confidence * 100)}% confidence` : '';
          const src = `${entityCount} source${entityCount !== 1 ? 's' : ''}`;
          sourceChip = [src, pct].filter(Boolean).join(' · ');
        }

        const reasoningEdges = result.info?.entities?.reasoning_edges || [];
        const sourceEntities = result.info?.entities?.used_entities || [];

        addMessage({
          type: 'ai',
          message: cleanedText,
          ts: Date.now(),
          sections,
          sourceChip,
          reasoningEdges,
          sourceEntities,
        });
      } catch (error) {
        const err = error as Error;
        const msg =
          err.name === 'AbortError'
            ? 'Request timed out — the model took too long. Try a faster model or a shorter question.'
            : `Error: ${err.message}`;
        addMessage({ type: 'error', message: msg, ts: Date.now() });
      }
    },
    [sendQuestion, state.currentKGName, ragModelHook.vendor, ragModelHook.selectedModel, dispatch, addMessage],
  );

  const isEmpty = messages.length === 0;

  // Find the index of the last AI message for the three-dots menu
  let lastAiIndex = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].type === 'ai' && messages[i].sections) {
      lastAiIndex = i;
      break;
    }
  }

  return (
    <div className="flex flex-col h-full bg-base-200">
      {/* Messages area */}
      <div ref={chatBoxRef} className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden px-3 py-2" aria-live="polite">
        {isEmpty ? (
          <ChatSuggestions onSelect={handleSend} />
        ) : (
          messages.map((msg, i) => {
            const msgKey = `${msg.ts ?? 0}-${i}`;
            if (msg.type === 'ai' && msg.sections) {
              const isLast = i === lastAiIndex;
              return (
                <div key={msgKey} className="chat chat-start">
                  <div className="chat-bubble rounded-2xl text-sm">
                    <ResponseSections
                      sections={msg.sections}
                      sourceChip={msg.sourceChip}
                      isLast={isLast}
                      onClearChat={handleClearChat}
                      onExportChat={handleExportChat}
                    />
                  </div>
                  <SourcesSection reasoningEdges={msg.reasoningEdges} sourceEntities={msg.sourceEntities} />
                  {msg.ts && (
                    <div className="chat-footer opacity-50 text-xs">{new Date(msg.ts).toLocaleTimeString()}</div>
                  )}
                </div>
              );
            }
            return (
              <ChatMessage
                key={msgKey}
                message={msg.message}
                type={msg.type}
                timestamp={msg.ts ? new Date(msg.ts).toLocaleTimeString() : undefined}
              />
            );
          })
        )}
        {sending && (
          <div className="chat chat-start">
            <div className="chat-bubble chat-bubble-ghost">
              <span className="loading loading-dots loading-xs" />
            </div>
          </div>
        )}
      </div>
      {/* Chat input at bottom */}
      <div className="shrink-0 px-3 pb-3 pt-1">
        <ChatInput onSend={handleSend} disabled={sending} ragModelHook={ragModelHook} />
      </div>
    </div>
  );
}
