import { api } from './api';

const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

function okResponse(data: unknown) {
  return { ok: true, status: 200, json: () => Promise.resolve(data) };
}

function errorResponse(detail?: string) {
  return {
    ok: false,
    status: 400,
    json: () => Promise.resolve(detail ? { detail } : {}),
  };
}

describe('api', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  describe('fetchModels', () => {
    it('calls GET /models/:vendor and returns parsed response', async () => {
      mockFetch.mockResolvedValue(okResponse({ models: ['gpt-4'] }));
      const result = await api.fetchModels('openai');
      expect(mockFetch).toHaveBeenCalledWith('/models/openai', undefined);
      expect(result).toEqual({ models: ['gpt-4'] });
    });
  });

  describe('fetchKGList', () => {
    it('calls GET /kg/list', async () => {
      mockFetch.mockResolvedValue(okResponse({ kgs: [] }));
      const result = await api.fetchKGList();
      expect(mockFetch).toHaveBeenCalledWith('/kg/list', undefined);
      expect(result).toEqual({ kgs: [] });
    });
  });

  describe('fetchDefaultCredentials', () => {
    it('calls GET /neo4j/default_credentials', async () => {
      mockFetch.mockResolvedValue(okResponse({ uri: 'bolt://localhost', user: 'neo4j' }));
      const result = await api.fetchDefaultCredentials();
      expect(mockFetch).toHaveBeenCalledWith('/neo4j/default_credentials', undefined);
      expect(result).toEqual({ uri: 'bolt://localhost', user: 'neo4j' });
    });
  });

  describe('checkHealth', () => {
    it('calls GET /doctor', async () => {
      mockFetch.mockResolvedValue(okResponse({ status: 'ok', checks: [] }));
      const result = await api.checkHealth();
      expect(mockFetch).toHaveBeenCalledWith('/doctor', undefined);
      expect(result).toEqual({ status: 'ok', checks: [] });
    });
  });

  describe('clearKG', () => {
    it('calls POST /clear_kg with JSON content-type', async () => {
      mockFetch.mockResolvedValue(okResponse({ message: 'cleared' }));
      await api.clearKG();
      expect(mockFetch).toHaveBeenCalledWith(
        '/clear_kg',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        }),
      );
    });
  });

  describe('createKG', () => {
    it('calls POST /create_ontology_guided_kg with FormData', async () => {
      const formData = new FormData();
      formData.append('file', new Blob(['test']), 'test.pdf');
      mockFetch.mockResolvedValue(okResponse({ kg_id: '123' }));
      await api.createKG(formData);
      expect(mockFetch).toHaveBeenCalledWith(
        '/create_ontology_guided_kg',
        expect.objectContaining({
          method: 'POST',
          body: formData,
        }),
      );
    });
  });

  describe('loadFromNeo4j', () => {
    it('calls POST /load_kg_from_neo4j with FormData', async () => {
      const formData = new FormData();
      mockFetch.mockResolvedValue(okResponse({ kg_id: '456', graph_data: { nodes: [], relationships: [] } }));
      await api.loadFromNeo4j(formData);
      expect(mockFetch).toHaveBeenCalledWith(
        '/load_kg_from_neo4j',
        expect.objectContaining({
          method: 'POST',
          body: formData,
        }),
      );
    });
  });

  describe('sendChat', () => {
    it('calls POST /chat with JSON body and signal', async () => {
      const payload = { question: 'What is X?', provider_rag: 'openai', model_rag: 'gpt-4' };
      const controller = new AbortController();
      mockFetch.mockResolvedValue(okResponse({ response: 'answer' }));
      await api.sendChat(payload, controller.signal);
      expect(mockFetch).toHaveBeenCalledWith(
        '/chat',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: controller.signal,
        }),
      );
    });
  });

  describe('error handling', () => {
    it('throws with detail message on non-ok response', async () => {
      mockFetch.mockResolvedValue(errorResponse('Not found'));
      await expect(api.checkHealth()).rejects.toThrow('Not found');
    });

    it('throws with fallback message when response body is not JSON', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: () => Promise.reject(new Error('parse error')),
      });
      await expect(api.checkHealth()).rejects.toThrow('Request failed: /doctor');
    });

    it('throws with fallback message when detail is missing', async () => {
      mockFetch.mockResolvedValue(errorResponse());
      await expect(api.fetchModels('test')).rejects.toThrow('Request failed: /models/test');
    });
  });
});
