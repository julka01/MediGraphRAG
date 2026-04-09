import { ChevronDownIcon, ChevronRightIcon, Squares2X2Icon } from '@heroicons/react/24/outline';
import clsx from 'clsx';
import { memo, useState } from 'react';
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

interface ResponseSectionsProps {
  sections: ResponseSectionsType;
  sourceChip?: string;
}

export const ResponseSections = memo(function ResponseSections({
  sections,
  sourceChip,
}: ResponseSectionsProps) {
  const [sourcesExpanded, setSourcesExpanded] = useState(false);
  const hasAnySections = sections.recommendation || sections.evidence || sections.nextSteps || sections.reasoning;

  return (
    <div>
      {sourceChip && (
        <div className="mt-2">
          <button
            type="button"
            onClick={() => setSourcesExpanded(!sourcesExpanded)}
            className={clsx(
              'inline-flex items-center gap-1.5 px-3 py-1 text-xs text-base-content/60 hover:text-base-content/80 transition-colors',
              sourcesExpanded ? 'rounded-t-full bg-base-content/10' : 'rounded-full bg-base-content/10',
            )}
          >
            <Squares2X2Icon className="size-3.5" aria-hidden="true" />
            {sourceChip}
          </button>
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
        className={clsx(
          'inline-flex items-center gap-1.5 px-3 py-1 text-xs text-base-content/60 hover:text-base-content/80 transition-colors',
          expanded ? 'rounded-t-full bg-base-content/10' : 'rounded-full bg-base-content/10',
        )}
        onClick={() => setExpanded(!expanded)}
      >
        <Squares2X2Icon className="size-3.5" aria-hidden="true" />
        Sources
      </button>
      {expanded && (
        <div className="bg-base-content/5 rounded-b-lg px-3 py-2 mt-0 space-y-1 text-xs">
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
