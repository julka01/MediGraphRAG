import { act, renderHook } from '@testing-library/react';
import { api } from '../api';
import { useChat } from './useChat';

vi.mock('../api', () => ({
  api: {
    sendChat: vi.fn(),
  },
}));

const mockSendChat = vi.mocked(api.sendChat);
const HISTORY_KEY = 'kg-chat-history';

describe('useChat', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  it('initializes with empty messages when no localStorage', () => {
    const { result } = renderHook(() => useChat());
    expect(result.current.messages).toEqual([]);
    expect(result.current.sending).toBe(false);
  });

  it('loads history from localStorage on init', () => {
    const history = [{ type: 'user' as const, message: 'hello', ts: 1000 }];
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));

    const { result } = renderHook(() => useChat());
    expect(result.current.messages).toEqual(history);
  });

  it('handles corrupt localStorage gracefully', () => {
    localStorage.setItem(HISTORY_KEY, 'not-json{{{');

    const { result } = renderHook(() => useChat());
    expect(result.current.messages).toEqual([]);
  });

  it('addMessage appends to messages and persists to localStorage', () => {
    const { result } = renderHook(() => useChat());

    act(() => {
      result.current.addMessage({ type: 'user', message: 'test message', ts: 1000 });
    });

    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].message).toBe('test message');

    const stored = JSON.parse(localStorage.getItem(HISTORY_KEY)!);
    expect(stored).toHaveLength(1);
    expect(stored[0].message).toBe('test message');
  });

  it('sendQuestion prepends user message and calls API with correct payload', async () => {
    const chatResponse = { response: 'AI answer' };
    mockSendChat.mockResolvedValue(chatResponse);

    const { result } = renderHook(() => useChat());

    let response: unknown;
    await act(async () => {
      response = await result.current.sendQuestion('What is X?', 'my-kg', 'openai', 'gpt-4');
    });

    expect(result.current.messages[0].type).toBe('user');
    expect(result.current.messages[0].message).toBe('You: What is X?');
    expect(mockSendChat).toHaveBeenCalledWith(
      { question: 'What is X?', provider_rag: 'openai', model_rag: 'gpt-4', kg_name: 'my-kg' },
      expect.any(AbortSignal),
    );
    expect(response).toEqual(chatResponse);
    expect(result.current.sending).toBe(false);
  });

  it('sendQuestion omits kg_name when kgName is null', async () => {
    mockSendChat.mockResolvedValue({ response: 'ok' });
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.sendQuestion('question', null, 'openai', 'gpt-4');
    });

    const payload = mockSendChat.mock.calls[0][0];
    expect(payload).not.toHaveProperty('kg_name');
  });

  it('clearChat empties messages and removes localStorage key', () => {
    localStorage.setItem(HISTORY_KEY, JSON.stringify([{ type: 'user', message: 'hi' }]));
    const { result } = renderHook(() => useChat());

    act(() => {
      result.current.clearChat();
    });

    expect(result.current.messages).toEqual([]);
    expect(localStorage.getItem(HISTORY_KEY)).toBeNull();
  });

  it('caps saved history at 60 messages', () => {
    const { result } = renderHook(() => useChat());

    act(() => {
      for (let i = 0; i < 65; i++) {
        result.current.addMessage({ type: 'user', message: `msg${i}`, ts: i });
      }
    });

    const stored = JSON.parse(localStorage.getItem(HISTORY_KEY)!);
    expect(stored.length).toBeLessThanOrEqual(60);
  });

  it('only saves user and ai message types to localStorage', () => {
    const { result } = renderHook(() => useChat());

    act(() => {
      result.current.addMessage({ type: 'user', message: 'question', ts: 1 });
      result.current.addMessage({ type: 'thinking', message: 'thinking...', ts: 2 });
      result.current.addMessage({ type: 'ai', message: 'answer', ts: 3 });
      result.current.addMessage({ type: 'error', message: 'oops', ts: 4 });
    });

    const stored = JSON.parse(localStorage.getItem(HISTORY_KEY)!);
    const types = stored.map((m: { type: string }) => m.type);
    expect(types).toEqual(['user', 'ai']);
  });
});
