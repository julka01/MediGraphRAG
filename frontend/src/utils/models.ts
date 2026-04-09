/** Strip vendor prefix (e.g. "openai/") and ":free" suffix for display. */
export function shortenModelName(name: string): string {
  if (!name) return '';
  let short = name;
  const slashIndex = short.indexOf('/');
  if (slashIndex !== -1) {
    short = short.slice(slashIndex + 1);
  }
  if (short.endsWith(':free')) {
    short = short.slice(0, -5);
  }
  return short;
}
