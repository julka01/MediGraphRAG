import { useState } from 'react';
import type { ReasoningEdge, ResponseSections as ResponseSectionsType, SourceEntity } from '../../types/app';
import { formatMarkdown, formatReasoningPath } from '../../utils/markdown';

interface SectionProps {
  title: string;
  content: string | undefined;
  defaultExpanded?: boolean;
  formatter?: (text: string) => string;
}

function Section({ title, content, defaultExpanded = false, formatter = formatMarkdown }: SectionProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  if (!content?.trim()) return null;

  return (
    <div className="mt-2">
      <button
        type="button"
        className="btn btn-ghost btn-xs text-xs font-semibold w-full justify-start"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? '▾' : '▸'} {title}
      </button>
      {expanded && (
        // biome-ignore lint/security/noDangerouslySetInnerHtml: content is sanitized markdown HTML from the server
        <div className="pl-4 text-sm mt-1" dangerouslySetInnerHTML={{ __html: formatter(content) }} />
      )}
    </div>
  );
}

interface ResponseSectionsProps {
  sections: ResponseSectionsType;
  sourceChip?: string;
}

export function ResponseSections({ sections, sourceChip }: ResponseSectionsProps) {
  const hasAnySections = sections.recommendation || sections.evidence || sections.nextSteps || sections.reasoning;

  return (
    <div>
      {sourceChip && <div className="badge badge-ghost badge-sm mb-2">◈ {sourceChip}</div>}
      {hasAnySections ? (
        <>
          <Section title="Summary" content={sections.recommendation} defaultExpanded={true} />
          <Section title="Evidence" content={sections.evidence} defaultExpanded={true} />
          <Section title="Next steps" content={sections.nextSteps} />
          <Section title="Reasoning path" content={sections.reasoning} formatter={formatReasoningPath} />
        </>
      ) : (
        // biome-ignore lint/security/noDangerouslySetInnerHtml: fallback is sanitized markdown HTML from the server
        <div className="text-sm" dangerouslySetInnerHTML={{ __html: formatMarkdown(sections.fallback || '') }} />
      )}
    </div>
  );
}

interface SourcesSectionProps {
  reasoningEdges?: ReasoningEdge[];
  sourceEntities?: SourceEntity[];
}

export function SourcesSection({ reasoningEdges, sourceEntities }: SourcesSectionProps) {
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
        {expanded ? '▾' : '▸'} Sources
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
}
