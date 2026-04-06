import type {
  ChatPayload,
  ChatResponse,
  ClearKGResponse,
  CreateKGResponse,
  DefaultCredentialsResponse,
  HealthResponse,
  KGListResponse,
  LoadNeo4jResponse,
  ModelsResponse,
} from './types/app';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error: { detail?: string } = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Request failed: ${url}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  fetchModels: (vendor: string) => request<ModelsResponse>(`/models/${vendor}`),
  fetchKGList: () => request<KGListResponse>('/kg/list'),
  fetchDefaultCredentials: () => request<DefaultCredentialsResponse>('/neo4j/default_credentials'),
  checkHealth: () => request<HealthResponse>('/doctor'),
  clearKG: () =>
    request<ClearKGResponse>('/clear_kg', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    }),
  createKG: (formData: FormData) =>
    request<CreateKGResponse>('/create_ontology_guided_kg', {
      method: 'POST',
      body: formData,
    }),
  loadFromNeo4j: (formData: FormData) =>
    request<LoadNeo4jResponse>('/load_kg_from_neo4j', {
      method: 'POST',
      body: formData,
    }),
  sendChat: (payload: ChatPayload, signal?: AbortSignal) =>
    request<ChatResponse>('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal,
    }),
};
