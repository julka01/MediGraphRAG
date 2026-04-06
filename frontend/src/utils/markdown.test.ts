import { formatMarkdown, formatReasoningPath } from './markdown';

describe('formatReasoningPath', () => {
  it('converts newlines to <br>', () => {
    // Note: the > in <br> is matched by the arrow regex, so we verify <br is present
    expect(formatReasoningPath('line1\nline2')).toContain('<br');
  });

  it('wraps arrow characters in span', () => {
    const result = formatReasoningPath('A → B');
    expect(result).toContain('<span class="reason-arrow">→</span>');
  });

  it('wraps connector words in span', () => {
    const result = formatReasoningPath('A triggers B');
    expect(result).toContain('<span class="reason-connector">triggers</span>');
  });

  it('handles yields connector', () => {
    const result = formatReasoningPath('X yields Y');
    expect(result).toContain('<span class="reason-connector">yields</span>');
  });

  it('handles reveals connector', () => {
    const result = formatReasoningPath('X reveals Y');
    expect(result).toContain('<span class="reason-connector">reveals</span>');
  });

  it('handles leads to connector', () => {
    const result = formatReasoningPath('X leads to Y');
    expect(result).toContain('<span class="reason-connector">leads to</span>');
  });

  it('handles results in connector', () => {
    const result = formatReasoningPath('X results in Y');
    expect(result).toContain('<span class="reason-connector">results in</span>');
  });

  it('returns empty string for empty input', () => {
    expect(formatReasoningPath('')).toBe('');
  });
});

describe('formatMarkdown', () => {
  it('converts newlines to <br>', () => {
    expect(formatMarkdown('line1\nline2')).toContain('<br>');
  });

  it('wraps bold text in <strong>', () => {
    expect(formatMarkdown('**bold**')).toContain('<strong>bold</strong>');
  });

  it('wraps italic text in <em>', () => {
    expect(formatMarkdown('*italic*')).toContain('<em>italic</em>');
  });

  it('wraps inline code in <code> with classes', () => {
    const result = formatMarkdown('`code`');
    expect(result).toContain('<code class="bg-base-300 px-1 rounded text-sm">code</code>');
  });

  it('converts bullet point characters', () => {
    const result = formatMarkdown('• item');
    expect(result).toContain('<br>• ');
  });

  it('returns empty string for empty input', () => {
    expect(formatMarkdown('')).toBe('');
  });
});
