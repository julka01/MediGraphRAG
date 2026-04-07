import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ChatMessage } from '../ChatMessage';

// navigator.clipboard is not available in jsdom — stub it so copy tests don't throw.
Object.defineProperty(navigator, 'clipboard', {
  value: { writeText: vi.fn().mockResolvedValue(undefined) },
  writable: true,
});

describe('ChatMessage', () => {
  it('renders user message with chat-end', () => {
    render(<ChatMessage message="hello" type="user" />);
    expect(screen.getByText('hello')).toBeInTheDocument();
    const chat = screen.getByText('hello').closest('.chat');
    expect(chat?.className).toContain('chat-end');
  });

  it('renders AI message with chat-start', () => {
    render(<ChatMessage message="AI response" type="ai" />);
    // react-markdown wraps text in a <p>; query by text content
    expect(screen.getByText('AI response')).toBeInTheDocument();
    const chat = screen.getByText('AI response').closest('.chat');
    expect(chat?.className).toContain('chat-start');
  });

  it('renders error message with error styling', () => {
    render(<ChatMessage message="Something went wrong" type="error" />);
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    const bubble = screen.getByText('Something went wrong').closest('.chat-bubble');
    expect(bubble?.className).toContain('chat-bubble-error');
  });

  it('renders thinking state with loading indicator', () => {
    render(<ChatMessage message="" type="thinking" />);
    const bubble = document.querySelector('.chat-bubble-ghost');
    expect(bubble).not.toBeNull();
  });

  it('shows copy button for AI messages', () => {
    render(<ChatMessage message="Copy me" type="ai" />);
    expect(screen.getByText('copy')).toBeInTheDocument();
  });

  it('renders timestamp when provided', () => {
    render(<ChatMessage message="hi" type="user" timestamp="10:30:00 AM" />);
    expect(screen.getByText('10:30:00 AM')).toBeInTheDocument();
  });
});
