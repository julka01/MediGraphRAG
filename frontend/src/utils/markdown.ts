export function formatReasoningPath(text: string): string {
  return text
    .replace(/\n/g, '<br>')
    .replace(/[→—>⇒]/g, '<span class="reason-arrow">→</span>')
    .replace(/\b(triggers|yields|reveals|leads to|results in)\b/gi, '<span class="reason-connector">$1</span>');
}

export function formatMarkdown(text: string): string {
  return text
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code class="bg-base-300 px-1 rounded text-sm">$1</code>')
    .replace(/[•·]/g, '<br>• ')
    .replace(/^[\s]*[•·]\s*/gm, '<li style="margin:6px 0;margin-left:18px;">')
    .replace(/(\n|^)(\d+)\.\s+/gm, '$1$2. ');
}
