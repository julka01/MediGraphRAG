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

function trustColor(val: number): string {
  if (val >= 0.7) return 'badge-success';
  if (val >= 0.4) return 'badge-warning';
  return 'badge-error';
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

  const trust = msg.trustSignals;
  const hasTrust =
    trust &&
    (trust.structural_support !== undefined || trust.grounding_support !== undefined || trust.confidence !== undefined);

  return (
    <div key={msgKey} className="chat chat-start">
      <div className="chat-bubble rounded-2xl border border-base-content/10 bg-base-100/82 text-sm shadow-sm">
        <ResponseSections sections={msg.sections} />
        {hasTrust && trust && (
          <div className="flex flex-wrap gap-1.5 mt-2 pt-2 border-t border-base-content/10">
            {trust.structural_support !== undefined && (
              <span
                className={`badge badge-xs gap-1 ${trustColor(trust.structural_support)}`}
                title="Graph-path support: fraction of answer entities reachable from question entities in the KG"
              >
                Structural {Math.round(trust.structural_support * 100)}%
              </span>
            )}
            {trust.grounding_support !== undefined && (
              <span
                className={`badge badge-xs gap-1 ${trustColor(trust.grounding_support)}`}
                title="Grounding: how well retrieved evidence supports the answer"
              >
                Grounding {Math.round(trust.grounding_support * 100)}%
              </span>
            )}
            {trust.confidence !== undefined && (
              <span
                className={`badge badge-xs gap-1 ${trustColor(trust.confidence)}`}
                title="Overall answer confidence"
              >
                Confidence {Math.round(trust.confidence * 100)}%
              </span>
            )}
          </div>
        )}
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
  const modelLabel =
    ragModelHook.models.find((model) => model.value === ragModelHook.selectedModel)?.label ||
    ragModelHook.selectedModel;

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
          const trustSignals = {
            structural_support: result.info?.structural_support,
            grounding_support: result.info?.grounding_support,
            confidence: result.info?.confidence,
          };

          addMessage({
            type: 'ai',
            message: cleanedText,
            ts: Date.now(),
            sections,
            reasoningEdges,
            sourceEntities,
            entityNames: nodeNames,
            trustSignals,
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
    <div className="flex h-full flex-col border-l border-base-content/10 bg-base-200/55 backdrop-blur-xl">
      <div className="shrink-0 border-b border-base-content/10 bg-base-100/45 px-4 py-2 backdrop-blur-sm">
        <div className="flex flex-wrap justify-end gap-2 text-2xs">
          <span className="rounded-full border border-base-content/10 bg-base-100/70 px-2.5 py-1 text-base-content/65">
            {state.currentKGName ? `KG · ${state.currentKGName}` : 'No KG loaded'}
          </span>
          <span className="rounded-full border border-base-content/10 bg-base-100/70 px-2.5 py-1 text-base-content/65">
            {ragModelHook.vendor.toUpperCase()} · {modelLabel}
          </span>
        </div>
      </div>

      {/* Messages area */}
      <div ref={chatBoxRef} className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden px-4 py-3" aria-live="polite">
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
                <div className="chat-bubble flex items-center justify-center rounded-2xl border border-base-content/10 bg-base-100/82 text-sm shadow-sm">
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
                <div className="flex rounded-xl border border-base-content/10 bg-base-100/60 p-0.5 shadow-sm">
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
      <div className="shrink-0 border-t border-base-content/10 bg-base-100/35 px-4 pb-2 pt-2 backdrop-blur-sm">
        <ChatInput onSend={handleSend} disabled={sending} ragModelHook={ragModelHook} />
        <div className="flex items-center justify-center gap-3 mt-1 text-2xs text-base-content/30">
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
