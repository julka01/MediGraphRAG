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
  afterEach(() => {
    delete document.body.dataset.theme;
  });

  it('returns light theme colors by default', () => {
    document.body.dataset.theme = 'light';
    const theme = getGraphTheme();
    expect(theme.nodeText).toBe('#1a1a1a');
    expect(theme.nodeTextDimmed).toBe('#bbbbbb');
    expect(theme.edgeText).toBe('#555555');
    expect(theme.dimmedNodeBg).toBe('#d8dde4');
    expect(theme.dimmedNodeBdr).toBe('#c8cdd4');
    expect(theme.dimmedEdge).toBe('#dde0e4');
  });

  it('returns dark theme colors when data-theme is dark', () => {
    document.body.dataset.theme = 'dark';
    const theme = getGraphTheme();
    expect(theme.nodeText).toBe('#ffffff');
    expect(theme.nodeTextDimmed).toBe('#444444');
    expect(theme.edgeText).toBe('#888888');
    expect(theme.dimmedNodeBg).toBe('#2a2a2a');
    expect(theme.dimmedNodeBdr).toBe('#3a3a3a');
    expect(theme.dimmedEdge).toBe('#282828');
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
