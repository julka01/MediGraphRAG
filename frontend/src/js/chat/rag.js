import { state } from '../state.js';
import { api } from '../api.js';
import { addToChat } from './messages.js';
import { initializeGraph } from '../graph/network.js';
import { applyHighlightStyles, updateHighlightBadge } from '../graph/legend.js';

// Helper function to format reasoning path with arrows
function formatReasoningPath(text) {
  return text
    .replace(/\n/g, '<br>')
    .replace(/[→—>⇒]/g, '<span class="reason-arrow">→</span>')
    .replace(/\b(triggers|yields|reveals|leads to|results in)\b/gi, '<span class="reason-connector">$1</span>');
}

// Helper function to format markdown content
function formatMarkdown(text) {
  return text
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.*?)\*\*/g, '<strong class="md-strong">$1</strong>')
    .replace(/\*(.*?)\*/g,     '<em class="md-em">$1</em>')
    .replace(/`(.*?)`/g,       '<code class="md-code">$1</code>')
    .replace(/[•·]/g, '<br>• ')
    .replace(/^[\s]*[•·]\s*/gm, '<li style="margin:6px 0;margin-left:18px;">')
    .replace(/(\n|^)(\d+)\.\s+/gm, '$1$2. ');
}

export function initRAG() {
  window.sendQuestion = async function() {
    const input = document.getElementById('question');
    const sendButton = document.getElementById('send-btn');
    const chatBox = document.getElementById('chat-box');
    const question = input.value.trim();
    if (!question) return;

    const vendor = document.getElementById('rag-vendor').value;
    const model = document.getElementById('rag-model').value;

    addToChat(`You: ${question}`, 'user');
    input.value = '';

    // Show thinking indicator
    const thinkingEl = addToChat('<span class="btn-dots"><span></span><span></span><span></span></span>', 'thinking', 'chat-thinking');

    const originalButtonText = sendButton.textContent;
    try {
      sendButton.disabled = true;

      // Create payload with required parameters
      const payload = {
        question: question,
        provider_rag: vendor,
        model_rag: model
      };

      // Add kg_name if available to filter retrieval to the selected KG
      if (state.currentKGName) {
        payload.kg_name = state.currentKGName;
      }

      const chatAbort = new AbortController();
      const chatTimeout = setTimeout(() => chatAbort.abort(), 130000); // 130 s (server timeout is 120 s)
      let result;
      try {
        result = await api.sendChat(payload, chatAbort.signal);
      } finally {
        clearTimeout(chatTimeout);
      }

      // Update highlighted nodes from used_entities in response.
      // entity.description = human-readable name (matches p.name on graph nodes).
      // entity.id = UUID-prefixed key (matches p.id on graph nodes as fallback).
      // highlightedNodes is a lookup Set (may hold 2 keys per entity) — count
      // entities separately so the badge shows the real entity count.
      state.highlightedNodes.clear();
      const usedEntities = result.info?.entities?.used_entities || [];
      usedEntities.forEach(entity => {
        const readable = (entity.description || '').toLowerCase().trim();
        const idKey    = (entity.id || '').toLowerCase().trim();
        if (readable) state.highlightedNodes.add(readable);
        if (idKey)    state.highlightedNodes.add(idKey);
      });
      updateHighlightBadge(usedEntities.length);

      // Apply highlight styles in-place to preserve layout; fall back to full
      // re-initialization only if the network hasn't been created yet.
      if (state.network) {
        applyHighlightStyles();
      } else if (state.graphData) {
        initializeGraph(state.graphData);
      }

      // Enhanced response formatting for new RAG structured format
      let formattedResponse = result.response || result.message || 'No response generated';

      // Parse the structured response into sections (simple line-by-line parsing)
      let sections = {
        recommendation: '',
        reasoning: '',
        evidence: '',
        nextSteps: ''
      };

      const lines = formattedResponse.split('\n');
      let currentSection = '';
      let currentContent = [];

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();

        // Check for section headers (case-insensitive)
        if (line.toUpperCase().includes('RECOMMENDATION/SUMMARY') && line.includes('#')) {
          // Save previous section before starting new one
          if (currentSection && currentContent.length > 0) {
            sections[currentSection] = currentContent.join('\n').trim();
            currentContent = [];
          }
          currentSection = 'recommendation';
        } else if (line.toUpperCase().includes('REASONING PATH') && line.includes('#')) {
          // Save previous section
          if (currentSection && currentContent.length > 0) {
            sections[currentSection] = currentContent.join('\n').trim();
            currentContent = [];
          }
          currentSection = 'reasoning';
        } else if (line.toUpperCase().includes('COMBINED EVIDENCE') && line.includes('#')) {
          // Save previous section
          if (currentSection && currentContent.length > 0) {
            sections[currentSection] = currentContent.join('\n').trim();
            currentContent = [];
          }
          currentSection = 'evidence';
        } else if (line.toUpperCase().includes('NEXT STEPS') && line.includes('#')) {
          // Save previous section
          if (currentSection && currentContent.length > 0) {
            sections[currentSection] = currentContent.join('\n').trim();
            currentContent = [];
          }
          currentSection = 'nextSteps';
        } else if (currentSection && line) {
          // Add non-empty lines to current section content
          currentContent.push(line);
        }
      }

      // Save the last section
      if (currentSection && currentContent.length > 0) {
        sections[currentSection] = currentContent.join('\n').trim();
      }

      // Format sections in desired order: Summary, Combined Evidence, Next Steps, Reasoning Path (at bottom)
      let formattedSections = [];

      // Recommendation/Summary (first, when present)
      if (sections.recommendation.trim()) {
        formattedSections.push(`
          <div class="response-section recommendation-section">
            <button class="section-toggle expanded" onclick="toggleSection(this)">▾ Summary</button>
            <div class="section-content expanded">
              ${formatMarkdown(sections.recommendation)}
            </div>
          </div>
        `);
      }

      // Combined Evidence (always present, second)
      if (sections.evidence.trim()) {
        formattedSections.push(`
          <div class="response-section evidence-section">
            <button class="section-toggle expanded" onclick="toggleSection(this)">▾ Evidence</button>
            <div class="section-content expanded">
              ${formatMarkdown(sections.evidence)}
            </div>
          </div>
        `);
      }

      // Next Steps (third, when present)
      if (sections.nextSteps.trim()) {
        formattedSections.push(`
          <div class="response-section next-steps-section">
            <button class="section-toggle" onclick="toggleSection(this)">▸ Next steps</button>
            <div class="section-content collapsed">
              ${formatMarkdown(sections.nextSteps)}
            </div>
          </div>
        `);
      }

      // Reasoning Path (always present, at bottom, collapsible)
      if (sections.reasoning.trim()) {
        formattedSections.push(`
          <div class="response-section reasoning-section">
            <button class="section-toggle" onclick="toggleSection(this)">▸ Reasoning path</button>
            <div class="section-content collapsed">
              ${formatReasoningPath(sections.reasoning)}
            </div>
          </div>
        `);
      }

      // If no sections were parsed, fall back to original formatting
      if (formattedSections.length === 0) {
        formattedSections.push(`
          <div class="response-section full-response">
            ${formatMarkdown(formattedResponse)}
          </div>
        `);
      }

      formattedResponse = formattedSections.join('<br>');

      // Prepend source chip with confidence + entity count
      const confidence = result.info?.confidence_score;
      const entityCount = (result.info?.entities?.used_entities || []).length;
      if (entityCount > 0) {
        const pct = confidence !== undefined ? `${Math.round(confidence * 100)}% confidence` : '';
        const src = `${entityCount} source${entityCount !== 1 ? 's' : ''}`;
        const chipText = [src, pct].filter(Boolean).join(' · ');
        formattedResponse = `<div class="source-chip">◈ ${chipText}</div>` + formattedResponse;
      }

      // Build Sources panel from entity names + reasoning edges
      const reasoningEdges = result.info?.entities?.reasoning_edges || [];
      const sourceEntities = result.info?.entities?.used_entities || [];
      if (reasoningEdges.length > 0 || sourceEntities.length > 0) {
        let sourceLines = '';
        // Deduplicate edges by serialized key
        const seenEdges = new Set();
        reasoningEdges.forEach(edge => {
          const fromName = edge.from_name || edge.from || '?';
          const toName   = edge.to_name   || edge.to   || '?';
          const rel      = (edge.relationship || 'CONNECTED_TO').replace(/_/g, ' ');
          const key = `${fromName}|${rel}|${toName}`;
          if (!seenEdges.has(key)) {
            seenEdges.add(key);
            sourceLines += `<div class="src-edge"><span class="src-node">${fromName}</span><span class="src-rel"> ──${rel}──▶ </span><span class="src-node">${toName}</span></div>`;
          }
        });
        // If no edges, fall back to listing entity names
        if (!sourceLines) {
          const names = [...new Set(sourceEntities.map(e => e.description || e.id).filter(Boolean))];
          sourceLines = names.map(n => `<span class="src-node">${n}</span>`).join(', ');
        }
        formattedResponse += `
          <div class="response-section sources-section">
            <button class="section-toggle" onclick="toggleSection(this)">▸ Sources</button>
            <div class="section-content collapsed">
              <div class="src-edges-list">${sourceLines}</div>
            </div>
          </div>`;
      }

      // Strip residual [Source: ...] document citations from the response text
      formattedResponse = formattedResponse.replace(/【Source:[^】]*】/g, '').replace(/\[Source:[^\]]*\]/g, '');

      // Remove thinking indicator
      if (thinkingEl) thinkingEl.remove();
      addToChat(`${formattedResponse}`, 'ai');
    } catch (error) {
      if (thinkingEl) thinkingEl.remove();
      const msg = error.name === 'AbortError'
        ? 'Request timed out — the model took too long. Try a faster model or a shorter question.'
        : `Error: ${error.message}`;
      addToChat(msg, 'error');
    } finally {
      // Restore button state
      sendButton.innerHTML = originalButtonText;
      sendButton.disabled = false;
    }
  };

  document.getElementById('question').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      window.sendQuestion();
    }
  });
}
