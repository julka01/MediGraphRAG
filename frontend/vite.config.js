import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/models': 'http://localhost:8000',
      '/kg': 'http://localhost:8000',
      '/chat': 'http://localhost:8000',
      '/doctor': 'http://localhost:8000',
      '/neo4j': 'http://localhost:8000',
      '/create_ontology_guided_kg': 'http://localhost:8000',
      '/load_kg_from_neo4j': 'http://localhost:8000',
      '/clear_kg': 'http://localhost:8000',
      '/kg_progress_stream': 'http://localhost:8000',
      '/static': 'http://localhost:8000',
    },
  },
});
