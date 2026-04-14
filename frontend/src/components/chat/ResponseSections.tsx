import { SparklesIcon } from '@heroicons/react/24/outline';
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
  if (!content?.trim()) return null;

  const formatted = formatter ? formatter(content) : content;

  return (
    <div className="collapse collapse-arrow mt-2 bg-transparent">
      <input type="checkbox" defaultChecked={defaultExpanded} aria-label={`Toggle ${title}`} />
      <div className="collapse-title text-xs font-semibold px-0 min-h-0 py-1 after:text-base-content/40">
        {title}
      </div>
      <div className="collapse-content text-sm px-0">
        <Markdown remarkPlugins={[remarkGfm]}>{formatted}</Markdown>
      </div>
    </div>
  );
}

interface ResponseSectionsProps {
  sections: ResponseSectionsType;
}

export const ResponseSections = memo(function ResponseSections({ sections }: ResponseSectionsProps) {
  const hasAnySections = sections.recommendation || sections.evidence || sections.nextSteps || sections.reasoning;

  return (
    <div>
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
  onHighlight?: () => void;
  isHighlighted?: boolean;
}

const INITIAL_VISIBLE = 4;

export const SourcesSection = memo(function SourcesSection({
  reasoningEdges,
  sourceEntities,
  onHighlight,
  isHighlighted,
}: SourcesSectionProps) {
  const [showAll, setShowAll] = useState(false);
  if (!reasoningEdges?.length && !sourceEntities?.length) return null;

  const seenEdges = new Set<string>();
  const uniqueEdges = (reasoningEdges || []).filter((edge) => {
    const key = `${edge.from_name || edge.from}|${edge.relationship}|${edge.to_name || edge.to}`;
    if (seenEdges.has(key)) return false;
    seenEdges.add(key);
    return true;
  });
  const uniqueEntities = [...new Set((sourceEntities || []).map((e) => e.description || e.id).filter(Boolean))];
  const hasEdges = uniqueEdges.length > 0;
  const count = hasEdges ? uniqueEdges.length : uniqueEntities.length;
  const visibleEdges = showAll ? uniqueEdges : uniqueEdges.slice(0, INITIAL_VISIBLE);
  const overflow = hasEdges ? uniqueEdges.length - INITIAL_VISIBLE : 0;

  return (
    <div className="mt-3 pt-2 border-t border-base-content/5">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-2xs uppercase tracking-wider text-base-content/40 ml-2">
          {count} {count === 1 ? 'Source' : 'Sources'}
        </span>
        {onHighlight && (
          <button
            type="button"
            className={`inline-flex items-center gap-1 px-2 py-0.5 mr-2 rounded-full text-2xs transition-colors ${
              isHighlighted
                ? 'bg-graph-highlight/15 text-graph-highlight'
                : 'text-base-content/40 hover:text-base-content/60'
            }`}
            onClick={onHighlight}
          >
            <SparklesIcon className="size-3" aria-hidden="true" />
            Highlight
          </button>
        )}
      </div>
      {hasEdges ? (
        <div className="space-y-1">
          {visibleEdges.map((edge) => (
            <div
              key={`${edge.from_name || edge.from}|${edge.relationship}|${edge.to_name || edge.to}`}
              className="flex items-center gap-1.5 px-2 py-1 rounded bg-base-content/5 text-xs min-w-0"
            >
              <span className="font-medium truncate">{edge.from_name || edge.from || '?'}</span>
              <span className="text-base-content/40 shrink-0 text-2xs">
                {(edge.relationship || 'CONNECTED_TO').replace(/_/g, ' ').toLowerCase()} →
              </span>
              <span className="font-medium truncate">{edge.to_name || edge.to || '?'}</span>
            </div>
          ))}
          {overflow > 0 && (
            <button
              type="button"
              className="text-2xs text-base-content/40 hover:text-base-content/60 transition-colors w-full text-right mt-0.5"
              onClick={() => setShowAll(!showAll)}
            >
              {showAll ? 'Show less' : `Show ${overflow} more`}
            </button>
          )}
        </div>
      ) : (
        <div className="flex flex-wrap gap-1">
          {uniqueEntities.map((name) => (
            <span key={name} className="px-2 py-0.5 rounded bg-base-content/5 text-xs font-medium">
              {name}
            </span>
          ))}
        </div>
      )}
    </div>
  );
});
