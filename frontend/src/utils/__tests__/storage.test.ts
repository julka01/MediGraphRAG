import { afterEach, describe, expect, it, vi } from 'vitest';
import { safeGet, safeSet } from '../storage';

describe('safeGet', () => {
  afterEach(() => localStorage.clear());

  it('returns stored value', () => {
    localStorage.setItem('key', 'value');
    expect(safeGet('key')).toBe('value');
  });

  it('returns null for missing key', () => {
    expect(safeGet('missing')).toBeNull();
  });

  it('returns fallback for missing key when provided', () => {
    expect(safeGet('missing', 'default')).toBe('default');
  });

  it('returns fallback when localStorage throws', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('quota exceeded');
    });
    expect(safeGet('key', 'fallback')).toBe('fallback');
    vi.restoreAllMocks();
  });
});

describe('safeSet', () => {
  afterEach(() => localStorage.clear());

  it('stores a value', () => {
    safeSet('key', 'value');
    expect(localStorage.getItem('key')).toBe('value');
  });

  it('does not throw when localStorage throws', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('quota exceeded');
    });
    expect(() => safeSet('key', 'value')).not.toThrow();
    vi.restoreAllMocks();
  });
});
