const SUGGESTIONS = [
  'What are the main entities in this knowledge graph?',
  'What relationships connect the key concepts?',
  'Summarize the most important findings.',
  'What evidence supports the central claims?',
] as const;

interface ChatSuggestionsProps {
  onSelect: (text: string) => void;
}

export function ChatSuggestions({ onSelect }: ChatSuggestionsProps) {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="flex flex-col gap-2 max-w-sm">
        {SUGGESTIONS.map((text) => (
          <button
            type="button"
            key={text}
            className="text-center px-4 py-2.5 rounded-2xl border border-base-content/20 text-sm text-base-content/50 hover:border-primary/50 hover:text-base-content/70 transition-colors"
            onClick={() => onSelect(text)}
          >
            {text}
          </button>
        ))}
      </div>
    </div>
  );
}
