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
    <div className="flex flex-col gap-2 px-2 py-4">
      {SUGGESTIONS.map((text) => (
        <button
          type="button"
          key={text}
          className="text-left px-4 py-2.5 rounded-2xl border border-base-300 text-sm text-base-content/50 hover:border-base-content/30 hover:text-base-content/70 transition-colors"
          onClick={() => onSelect(text)}
        >
          {text}
        </button>
      ))}
    </div>
  );
}
