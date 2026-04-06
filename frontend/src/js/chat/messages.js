import { state } from '../state.js';
import { showError } from '../ui/notifications.js';

/**
 * Adds a message to the chat box and optionally persists it to localStorage.
 *
 * @param {string} message - The message content (HTML for ai/thinking, plain text for user/error)
 * @param {string} type - One of 'user', 'ai', 'thinking', 'error'
 * @param {string|null} id - Optional DOM id for the message container
 * @param {boolean} skipHistory - If true, skip persisting to localStorage
 * @returns {HTMLElement} The message container element
 */
export function addToChat(message, type, id = null, skipHistory = false) {
    const chatBox = document.getElementById('chat-box');

    // Hide suggested questions on first real message
    const emptyState = document.getElementById('chat-empty-state');
    if (emptyState) emptyState.style.display = 'none';

    // Create message container with flexbox layout
    const messageContainer = document.createElement('div');
    messageContainer.className = 'message-container ' + type;

    const messageBubble = document.createElement('div');
    messageBubble.className = type + '-message';

    // Create internal bubble content
    const bubbleContent = document.createElement('div');
    bubbleContent.className = 'message-bubble';

    // User and error messages are plain text — set via textContent to prevent XSS.
    // AI and thinking messages contain intentional HTML formatting — use innerHTML.
    if (type === 'user' || type === 'error') {
        bubbleContent.textContent = message;
    } else {
        if (type === 'thinking') {
            message = `<strong>Agent:</strong> ${message}`;
        }
        bubbleContent.innerHTML = message;
    }
    messageBubble.appendChild(bubbleContent);

    // Add copy button for AI responses
    if (type === 'ai') {
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-msg-btn';
        copyBtn.title = 'Copy to clipboard';
        copyBtn.textContent = 'copy';
        copyBtn.addEventListener('click', function() {
            const text = messageBubble.innerText.replace(/^copy\s*/, '');
            navigator.clipboard.writeText(text).then(() => {
                copyBtn.textContent = 'copied!';
                setTimeout(() => { copyBtn.textContent = 'copy'; }, 1500);
            });
        });
        messageBubble.style.position = 'relative';
        messageBubble.appendChild(copyBtn);
    }

    // Add timestamp
    const timestampEl = document.createElement('div');
    timestampEl.className = 'message-timestamp';
    const now = new Date();
    timestampEl.textContent = now.toLocaleTimeString();
    messageBubble.appendChild(timestampEl);

    messageContainer.appendChild(messageBubble);

    if (id) {
        messageContainer.id = id;
    }

    chatBox.appendChild(messageContainer);
    // Only auto-scroll if user is already near the bottom (within 80px)
    const distFromBottom = chatBox.scrollHeight - chatBox.scrollTop - chatBox.clientHeight;
    if (distFromBottom < 80) chatBox.scrollTop = chatBox.scrollHeight;

    // Persist chat history to localStorage (only user + ai messages, skip thinking/error)
    if (!skipHistory && (type === 'user' || type === 'ai')) {
        try {
            const history = JSON.parse(localStorage.getItem('kg-chat-history') || '[]');
            history.push({ type, message, ts: Date.now() });
            // Keep last 60 messages
            if (history.length > 60) history.splice(0, history.length - 60);
            localStorage.setItem('kg-chat-history', JSON.stringify(history));
        } catch (e) { /* quota exceeded or unavailable */ }
    }

    return messageContainer;
}

/**
 * Initializes chat-related DOM wiring: clears chat, restores history,
 * and binds clear/export button listeners.
 */
export function initChat() {
    // Clear chat on page load
    const chatBox = document.getElementById('chat-box');
    if (chatBox) chatBox.innerHTML = '';

    // Setup error popup close button
    document.getElementById('close-error-popup').addEventListener('click', () => {
        document.getElementById('error-popup').classList.add('hidden');
    });

    // Restore chat history from localStorage
    try {
        const history = JSON.parse(localStorage.getItem('kg-chat-history') || '[]');
        if (history.length > 0) {
            const emptyState = document.getElementById('chat-empty-state');
            if (emptyState) emptyState.style.display = 'none';
            history.forEach(entry => {
                addToChat(entry.message, entry.type, null, true);
            });
        }
    } catch (_e) { /* ignore */ }

    // Clear chat button
    document.getElementById('clear-chat-btn').addEventListener('click', function() {
        const chatBox = document.getElementById('chat-box');
        chatBox.innerHTML = '';
        // Restore empty state
        const emptyState = document.getElementById('chat-empty-state');
        if (emptyState) {
            chatBox.appendChild(emptyState);
            emptyState.style.display = '';
        }
        localStorage.removeItem('kg-chat-history');
    });

    // Export chat as markdown
    document.getElementById('export-chat-btn').addEventListener('click', function() {
        const chatBox = document.getElementById('chat-box');
        const containers = chatBox.querySelectorAll('.message-container');
        if (!containers.length) { showError('No messages to export'); return; }
        let md = `# Chat Export — ${state.currentKGName || 'Knowledge Graph'}\n_Exported ${new Date().toLocaleString()}_\n\n`;
        containers.forEach(container => {
            const bubble = container.querySelector('.message-bubble');
            if (!bubble) return;
            const isUser = container.classList.contains('user');
            const isAI = container.classList.contains('ai');
            if (!isUser && !isAI) return;
            const clone = bubble.cloneNode(true);
            clone.querySelectorAll('.copy-msg-btn, .message-timestamp').forEach(el => el.remove());
            const text = (clone.innerText || clone.textContent || '').trim();
            if (!text) return;
            md += isUser ? `**You:** ${text}\n\n` : `**Assistant:** ${text}\n\n---\n\n`;
        });
        const blob = new Blob([md], { type: 'text/markdown' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `chat_${(state.currentKGName || 'export').replace(/\s+/g,'_')}_${new Date().toISOString().slice(0,10)}.md`;
        a.click();
        URL.revokeObjectURL(a.href);
    });
}
