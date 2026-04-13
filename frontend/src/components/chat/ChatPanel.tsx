import { memo, useCallback, useEffect, useRef } from 'react';
import { useApp } from '../../context/AppContext';
import { useChat } from '../../hooks/useChat';
import type {
  AppAction,
  ChatMessage as ChatMessageType,
  ResponseSections as ResponseSectionsType,
  UseModelsReturn,
} from '../../types/app';
import { showError } from '../ui/Notifications';
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

interface AiMessageRowProps {
  msg: ChatMessageType;
  msgKey: string;
  highlightedNodes: Set<string>;
  dispatch: React.Dispatch<AppAction>;
}

const AiMessageRow = memo(function AiMessageRow({ msg, msgKey, highlightedNodes, dispatch }: AiMessageRowProps) {
  if (!msg.sections) return null;
  const entityNames = msg.entityNames ?? new Set<string>();
  const hasEntities = entityNames.size > 0;
  const isMsgHighlighted =
    hasEntities && entityNames.size === highlightedNodes.size && [...entityNames].every((n) => highlightedNodes.has(n));

  const handleHighlight = hasEntities
    ? () => {
        if (isMsgHighlighted) {
          dispatch({ type: 'CLEAR_HIGHLIGHTED_NODES' });
        } else {
          dispatch({ type: 'SET_HIGHLIGHTED_NODES', nodes: entityNames });
        }
      }
    : undefined;

  return (
    <div key={msgKey} className="chat chat-start">
      <div className="chat-bubble rounded-2xl text-sm">
        <ResponseSections sections={msg.sections} />
        <SourcesSection
          reasoningEdges={msg.reasoningEdges}
          sourceEntities={msg.sourceEntities}
          isHighlighted={isMsgHighlighted}
          onHighlight={handleHighlight}
        />
      </div>
      {msg.ts && (
        <div className="chat-footer opacity-50 text-xs">
          {new Date(msg.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })}
        </div>
      )}
    </div>
  );
});

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
    (question: string): boolean => {
      if (!state.currentKGId) {
        showError(dispatch, 'Load a knowledge graph first');
        return false;
      }
      void (async () => {
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

          const reasoningEdges = result.info?.entities?.reasoning_edges || [];
          const sourceEntities = result.info?.entities?.used_entities || [];

          addMessage({
            type: 'ai',
            message: cleanedText,
            ts: Date.now(),
            sections,
            reasoningEdges,
            sourceEntities,
            entityNames: nodeNames,
          });
        } catch (error) {
          const err = error as Error;
          const msg =
            err.name === 'AbortError'
              ? 'Request timed out — the model took too long. Try a faster model or a shorter question.'
              : `Error: ${err.message}`;
          addMessage({ type: 'error', message: msg, ts: Date.now() });
        }
      })();
      return true;
    },
    [
      sendQuestion,
      state.currentKGId,
      state.currentKGName,
      ragModelHook.vendor,
      ragModelHook.selectedModel,
      dispatch,
      addMessage,
    ],
  );

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-full bg-base-200">
      {/* Messages area */}
      <div ref={chatBoxRef} className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden px-3 py-2" aria-live="polite">
        {isEmpty ? (
          <ChatSuggestions onSelect={handleSend} />
        ) : (
          <>
            {messages.map((msg, i) => {
              const msgKey = `${msg.ts ?? 0}-${i}`;
              if (msg.type === 'ai' && msg.sections) {
                return (
                  <AiMessageRow
                    key={msgKey}
                    msg={msg}
                    msgKey={msgKey}
                    highlightedNodes={state.highlightedNodes}
                    dispatch={dispatch}
                  />
                );
              }
              return (
                <ChatMessage
                  key={msgKey}
                  message={msg.message}
                  type={msg.type}
                  timestamp={
                    msg.ts
                      ? new Date(msg.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })
                      : undefined
                  }
                />
              );
            })}
            {sending && (
              <div className="chat chat-start">
                <div className="chat-bubble rounded-2xl bg-base-300 text-sm flex items-center justify-center">
                  <span
                    className="loading loading-dots loading-xs text-base-content"
                    aria-label="AI is thinking"
                    role="status"
                  />
                </div>
              </div>
            )}
            {messages.length > 0 && (
              <div className="flex justify-center py-2">
                <div className="flex bg-base-300/30 rounded-lg p-0.5">
                  <button
                    type="button"
                    className="px-2.5 py-1 text-2xs text-base-content/50 hover:text-base-content/80 rounded transition-colors"
                    onClick={handleClearChat}
                  >
                    Clear
                  </button>
                  <button
                    type="button"
                    className="px-2.5 py-1 text-2xs text-base-content/50 hover:text-base-content/80 rounded transition-colors"
                    onClick={handleExportChat}
                  >
                    Export
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
      {/* Chat input at bottom */}
      <div className="shrink-0 px-3 pb-1.5 pt-1">
        <ChatInput onSend={handleSend} disabled={sending} ragModelHook={ragModelHook} />
        <div className="flex items-center justify-end gap-3 mt-1 text-2xs text-base-content/30">
          <span className="flex items-center gap-1">
            <kbd className="kbd kbd-xs">&#x23CE;</kbd> send
          </span>
          <span className="flex items-center gap-1">
            <kbd className="kbd kbd-xs">&#x21E7;</kbd>
            <kbd className="kbd kbd-xs">&#x23CE;</kbd> new line
          </span>
        </div>
      </div>
    </div>
  );
}
