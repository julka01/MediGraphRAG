import { describe, expect, it } from 'vitest';
import { shortenModelName } from '../models';

describe('shortenModelName', () => {
  it('strips vendor prefix and :free suffix', () => {
    expect(shortenModelName('openai/gpt-oss-120b:free')).toBe('gpt-oss-120b');
  });

  it('strips only vendor prefix when no :free suffix', () => {
    expect(shortenModelName('meta-llama/llama-3.3-8b-instruct:free')).toBe('llama-3.3-8b-instruct');
  });

  it('strips deepseek prefix', () => {
    expect(shortenModelName('deepseek/deepseek-chat-v3.1:free')).toBe('deepseek-chat-v3.1');
  });

  it('strips x-ai prefix', () => {
    expect(shortenModelName('x-ai/grok-4-fast:free')).toBe('grok-4-fast');
  });

  it('returns as-is when no prefix or suffix', () => {
    expect(shortenModelName('gpt-4o')).toBe('gpt-4o');
  });

  it('handles empty string', () => {
    expect(shortenModelName('')).toBe('');
  });
});
