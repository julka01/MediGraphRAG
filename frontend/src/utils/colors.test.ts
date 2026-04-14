import { generateNodeTypeColors, generateRelationshipTypeColors } from './colors';

describe('generateNodeTypeColors', () => {
  it('maps types to palette colors by index', () => {
    const result = generateNodeTypeColors(['Disease', 'Drug']);
    expect(result).toEqual({
      Disease: '#a6cee3',
      Drug: '#1f78b4',
    });
  });

  it('wraps around when types exceed palette length', () => {
    const types = Array.from({ length: 25 }, (_, i) => `Type${i}`);
    const result = generateNodeTypeColors(types);
    expect(result.Type0).toBe('#a6cee3');
    expect(result.Type24).toBe('#a6cee3');
  });

  it('returns empty object for empty input', () => {
    expect(generateNodeTypeColors([])).toEqual({});
  });
});

describe('generateRelationshipTypeColors', () => {
  it('maps types to edge palette colors by index', () => {
    const result = generateRelationshipTypeColors(['TREATS', 'CAUSES']);
    expect(result).toEqual({
      TREATS: '#444444',
      CAUSES: '#666666',
    });
  });

  it('wraps around when types exceed palette length', () => {
    const types = Array.from({ length: 11 }, (_, i) => `Rel${i}`);
    const result = generateRelationshipTypeColors(types);
    expect(result.Rel0).toBe('#444444');
    expect(result.Rel10).toBe('#444444');
  });

  it('returns empty object for empty input', () => {
    expect(generateRelationshipTypeColors([])).toEqual({});
  });
});
