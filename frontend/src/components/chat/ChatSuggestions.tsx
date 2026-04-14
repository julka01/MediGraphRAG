interface ChatSuggestionsProps {
  onSelect: (text: string) => void;
}

export function ChatSuggestions({ onSelect }: ChatSuggestionsProps) {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="mx-auto flex max-w-md flex-col items-center text-center">
        <div className="panel-glass mb-5 flex size-14 items-center justify-center rounded-2xl text-xl text-primary">
          ◇
        </div>
        <p className="text-[0.65rem] font-medium uppercase tracking-[0.22em] text-base-content/45">Evidence chat</p>
        <h3 className="mt-2 text-lg font-semibold text-base-content">Ask grounded questions about the graph</h3>
        <p className="mt-2 max-w-sm text-sm leading-6 text-base-content/58">
          Load a graph, then ask for grounded answers, trace reasoning paths, or inspect supporting evidence.
        </p>
        <p className="mt-6 rounded-2xl border border-dashed border-base-content/12 bg-base-100/35 px-4 py-3 text-sm text-base-content/46">
          Start typing in the composer below.
        </p>
      </div>
    </div>
  );
}
