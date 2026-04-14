import { confidenceEdgeColor, getGraphTheme, normName } from './graph-helpers';

describe('confidenceEdgeColor', () => {
  it('returns gold for confidence >= 0.8', () => {
    expect(confidenceEdgeColor(0.8)).toBe('#f1c40f');
    expect(confidenceEdgeColor(1.0)).toBe('#f1c40f');
    expect(confidenceEdgeColor(0.95)).toBe('#f1c40f');
  });

  it('returns orange for confidence >= 0.5 and < 0.8', () => {
    expect(confidenceEdgeColor(0.5)).toBe('#e67e22');
    expect(confidenceEdgeColor(0.7)).toBe('#e67e22');
    expect(confidenceEdgeColor(0.79)).toBe('#e67e22');
  });

  it('returns gray for confidence >= 0.3 and < 0.5', () => {
    expect(confidenceEdgeColor(0.3)).toBe('#95a5a6');
    expect(confidenceEdgeColor(0.4)).toBe('#95a5a6');
    expect(confidenceEdgeColor(0.49)).toBe('#95a5a6');
  });

  it('returns dark gray for confidence below 0.3', () => {
    expect(confidenceEdgeColor(0.1)).toBe('#555e68');
    expect(confidenceEdgeColor(0)).toBe('#555e68');
    expect(confidenceEdgeColor(0.29)).toBe('#555e68');
  });

  it('returns null for null', () => {
    expect(confidenceEdgeColor(null)).toBeNull();
  });

  it('returns null for undefined', () => {
    expect(confidenceEdgeColor(undefined)).toBeNull();
  });
});

describe('getGraphTheme', () => {
  // getGraphTheme reads CSS custom properties via getComputedStyle. In jsdom,
  // CSS vars are not loaded from stylesheets, so readCSSColor returns the
  // hardcoded fallback values. Light/dark switching is handled purely by CSS
  // ([data-theme="light"] overrides), not by JS branching.

  it('returns all required theme keys', () => {
    const theme = getGraphTheme();
    expect(theme).toHaveProperty('nodeText');
    expect(theme).toHaveProperty('nodeTextDimmed');
    expect(theme).toHaveProperty('edgeText');
    expect(theme).toHaveProperty('edgeLabelBg');
    expect(theme).toHaveProperty('dimmedNodeBg');
    expect(theme).toHaveProperty('dimmedNodeBdr');
    expect(theme).toHaveProperty('dimmedEdge');
    expect(theme).toHaveProperty('highlight');
  });

  it('falls back to dark-theme defaults when CSS vars are not set', () => {
    const theme = getGraphTheme();
    // Fallback values match the dark-theme defaults defined in graph-helpers.ts
    expect(theme.nodeText).toBe('#ffffff');
    expect(theme.nodeTextDimmed).toBe('#444444');
    expect(theme.edgeText).toBe('#888888');
    expect(theme.dimmedNodeBg).toBe('#2a2a2a');
    expect(theme.dimmedNodeBdr).toBe('#3a3a3a');
    expect(theme.dimmedEdge).toBe('#282828');
    expect(theme.highlight).toBe('#ffd700');
  });

  it('uses CSS var value when set on documentElement', () => {
    document.documentElement.style.setProperty('--color-graph-node-text', '#123456');
    const theme = getGraphTheme();
    expect(theme.nodeText).toBe('#123456');
    document.documentElement.style.removeProperty('--color-graph-node-text');
  });
});

describe('normName', () => {
  it('lowercases text', () => {
    expect(normName('Hello')).toBe('hello');
  });

  it('replaces underscores with spaces', () => {
    expect(normName('hello_world')).toBe('hello world');
  });

  it('collapses multiple spaces', () => {
    expect(normName('hello   world')).toBe('hello world');
  });

  it('trims whitespace', () => {
    expect(normName('  hello  ')).toBe('hello');
  });

  it('handles mixed underscores and spaces', () => {
    expect(normName('Hello_World  Test')).toBe('hello world test');
  });

  it('returns empty string for null', () => {
    expect(normName(null)).toBe('');
  });

  it('returns empty string for undefined', () => {
    expect(normName(undefined)).toBe('');
  });

  it('returns empty string for empty string', () => {
    expect(normName('')).toBe('');
  });
});
