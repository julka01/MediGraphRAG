export function formatReasoningPath(text: string): string {
  return text.replace(/[→—>⇒]/g, ' → ').replace(/\b(triggers|yields|reveals|leads to|results in)\b/gi, '**$1**');
}
