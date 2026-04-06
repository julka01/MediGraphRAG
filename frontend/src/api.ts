async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Request failed: ${url}`);
  }
  return response.json();
}

export const api = {
  fetchModels: (vendor) => request(`/models/${vendor}`),
  fetchKGList: () => request('/kg/list'),
  fetchDefaultCredentials: () => request('/neo4j/default_credentials'),
  checkHealth: () => request('/doctor'),
  clearKG: () => request('/clear_kg', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  }),
  createKG: (formData) => request('/create_ontology_guided_kg', {
    method: 'POST',
    body: formData,
  }),
  loadFromNeo4j: (formData) => request('/load_kg_from_neo4j', {
    method: 'POST',
    body: formData,
  }),
  sendChat: (payload, signal) => request('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  }),
};
