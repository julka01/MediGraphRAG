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
    <div className="flex flex-col items-center justify-center h-full gap-4 px-4">
      <div className="text-sm opacity-60">Ask a question about the loaded knowledge graph</div>
      <div className="flex flex-wrap gap-2 justify-center">
        {SUGGESTIONS.map((text) => (
          <button key={text} className="btn btn-ghost btn-sm text-xs normal-case" onClick={() => onSelect(text)}>{text}</button>
        ))}
      </div>
    </div>
  );
}
