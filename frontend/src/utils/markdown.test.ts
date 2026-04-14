import { formatReasoningPath } from './markdown';

describe('formatReasoningPath', () => {
  it('normalises arrow characters to markdown arrow', () => {
    expect(formatReasoningPath('A → B')).toBe('A  →  B');
    expect(formatReasoningPath('A — B')).toBe('A  →  B');
    expect(formatReasoningPath('A > B')).toBe('A  →  B');
    expect(formatReasoningPath('A ⇒ B')).toBe('A  →  B');
  });

  it('wraps connector words in bold markdown', () => {
    expect(formatReasoningPath('A triggers B')).toContain('**triggers**');
  });

  it('handles yields connector', () => {
    expect(formatReasoningPath('X yields Y')).toContain('**yields**');
  });

  it('handles reveals connector', () => {
    expect(formatReasoningPath('X reveals Y')).toContain('**reveals**');
  });

  it('handles leads to connector', () => {
    expect(formatReasoningPath('X leads to Y')).toContain('**leads to**');
  });

  it('handles results in connector', () => {
    expect(formatReasoningPath('X results in Y')).toContain('**results in**');
  });

  it('is case-insensitive for connector words', () => {
    expect(formatReasoningPath('A Triggers B')).toContain('**Triggers**');
  });

  it('returns empty string unchanged for empty input', () => {
    expect(formatReasoningPath('')).toBe('');
  });

  it('does not produce HTML output', () => {
    const result = formatReasoningPath('A → B triggers C');
    expect(result).not.toContain('<');
    expect(result).not.toContain('>');
  });
});
