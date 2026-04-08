import { ChevronDownIcon, ChevronRightIcon, Squares2X2Icon } from '@heroicons/react/24/outline';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ReasoningEdge, ResponseSections as ResponseSectionsType, SourceEntity } from '../../types/app';
import { formatReasoningPath } from '../../utils/markdown';

interface SectionProps {
  title: string;
  content: string | undefined;
  defaultExpanded?: boolean;
  formatter?: (text: string) => string;
}

function Section({ title, content, defaultExpanded = false, formatter }: SectionProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  if (!content?.trim()) return null;

  const formatted = formatter ? formatter(content) : content;

  return (
    <div className="mt-2">
      <button
        type="button"
        className="btn btn-ghost btn-xs text-xs font-semibold w-full justify-start"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? (
          <ChevronDownIcon className="size-4 inline" aria-hidden="true" />
        ) : (
          <ChevronRightIcon className="size-4 inline" aria-hidden="true" />
        )}{' '}
        {title}
      </button>
      {expanded && (
        <div className="pl-4 text-sm mt-1">
          <Markdown remarkPlugins={[remarkGfm]}>{formatted}</Markdown>
        </div>
      )}
    </div>
  );
}

interface ThreeDotsMenuProps {
  onClearChat?: () => void;
  onExportChat?: () => void;
}

function ThreeDotsMenu({ onClearChat, onExportChat }: ThreeDotsMenuProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function handleClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [open]);

  const handleAction = useCallback(
    (action?: () => void) => {
      setOpen(false);
      action?.();
    },
    [],
  );

  return (
    <div ref={containerRef} className="flex justify-center mt-1 relative">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="btn btn-ghost btn-xs text-base-content/50 hover:text-base-content"
        aria-label="Message options"
      >
        &#x22EF;
      </button>
      {open && (
        <div className="absolute bottom-full mb-1 bg-base-100 border border-base-300 rounded-lg shadow-lg z-20 min-w-max">
          {onClearChat && (
            <button
              type="button"
              onClick={() => handleAction(onClearChat)}
              className="block w-full text-left px-3 py-1.5 text-xs hover:bg-base-200 transition-colors first:rounded-t-lg last:rounded-b-lg"
            >
              Clear Chat
            </button>
          )}
          {onExportChat && (
            <button
              type="button"
              onClick={() => handleAction(onExportChat)}
              className="block w-full text-left px-3 py-1.5 text-xs hover:bg-base-200 transition-colors first:rounded-t-lg last:rounded-b-lg"
            >
              Export Chat
            </button>
          )}
        </div>
      )}
    </div>
  );
}

interface ResponseSectionsProps {
  sections: ResponseSectionsType;
  sourceChip?: string;
  isLast?: boolean;
  onClearChat?: () => void;
  onExportChat?: () => void;
}

export const ResponseSections = memo(function ResponseSections({
  sections,
  sourceChip,
  isLast = false,
  onClearChat,
  onExportChat,
}: ResponseSectionsProps) {
  const hasAnySections = sections.recommendation || sections.evidence || sections.nextSteps || sections.reasoning;

  return (
    <div>
      {sourceChip && (
        <div className="btn btn-ghost btn-xs text-xs font-semibold w-full justify-start mb-2">
          <Squares2X2Icon className="size-4 inline" aria-hidden="true" /> {sourceChip}
        </div>
      )}
      {hasAnySections ? (
        <>
          <Section title="Summary" content={sections.recommendation} defaultExpanded={true} />
          <Section title="Evidence" content={sections.evidence} defaultExpanded={true} />
          <Section title="Next steps" content={sections.nextSteps} />
          <Section title="Reasoning path" content={sections.reasoning} formatter={formatReasoningPath} />
        </>
      ) : (
        <div className="text-sm">
          <Markdown remarkPlugins={[remarkGfm]}>{sections.fallback || ''}</Markdown>
        </div>
      )}
      {isLast && <ThreeDotsMenu onClearChat={onClearChat} onExportChat={onExportChat} />}
    </div>
  );
});

interface SourcesSectionProps {
  reasoningEdges?: ReasoningEdge[];
  sourceEntities?: SourceEntity[];
}

export const SourcesSection = memo(function SourcesSection({ reasoningEdges, sourceEntities }: SourcesSectionProps) {
  const [expanded, setExpanded] = useState(false);
  if (!reasoningEdges?.length && !sourceEntities?.length) return null;

  const seenEdges = new Set<string>();
  const uniqueEdges = (reasoningEdges || []).filter((edge) => {
    const key = `${edge.from_name || edge.from}|${edge.relationship}|${edge.to_name || edge.to}`;
    if (seenEdges.has(key)) return false;
    seenEdges.add(key);
    return true;
  });

  return (
    <div className="mt-2">
      <button
        type="button"
        className="btn btn-ghost btn-xs text-xs font-semibold w-full justify-start"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? (
          <ChevronDownIcon className="size-4 inline" aria-hidden="true" />
        ) : (
          <ChevronRightIcon className="size-4 inline" aria-hidden="true" />
        )}{' '}
        Sources
      </button>
      {expanded && (
        <div className="pl-4 mt-1 space-y-1 text-xs">
          {uniqueEdges.length > 0 ? (
            uniqueEdges.map((edge) => (
              <div
                key={`${edge.from_name || edge.from}|${edge.relationship}|${edge.to_name || edge.to}`}
                className="flex items-center gap-1 flex-wrap"
              >
                <span className="badge badge-xs badge-outline">{edge.from_name || edge.from || '?'}</span>
                <span className="opacity-50">──{(edge.relationship || 'CONNECTED_TO').replace(/_/g, ' ')}──▶</span>
                <span className="badge badge-xs badge-outline">{edge.to_name || edge.to || '?'}</span>
              </div>
            ))
          ) : (
            <div className="flex flex-wrap gap-1">
              {[...new Set((sourceEntities || []).map((e) => e.description || e.id).filter(Boolean))].map((name) => (
                <span key={name} className="badge badge-xs badge-outline">
                  {name}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
});
