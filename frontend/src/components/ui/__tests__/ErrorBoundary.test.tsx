import { render, screen } from '@testing-library/react';
import { ErrorBoundary } from '../ErrorBoundary';

function ThrowingComponent({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) throw new Error('Test error');
  return <div>Content works</div>;
}

describe('ErrorBoundary', () => {
  const originalError = console.error;
  beforeAll(() => {
    console.error = (...args: unknown[]) => {
      if (typeof args[0] === 'string' && args[0].includes('Error: Uncaught')) return;
      if (typeof args[0] === 'string' && args[0].includes('The above error')) return;
      originalError(...args);
    };
  });
  afterAll(() => {
    console.error = originalError;
  });

  it('renders children when no error', () => {
    render(
      <ErrorBoundary name="Test">
        <ThrowingComponent shouldThrow={false} />
      </ErrorBoundary>,
    );
    expect(screen.getByText('Content works')).toBeInTheDocument();
  });

  it('renders fallback when child throws', () => {
    render(
      <ErrorBoundary name="Test Panel">
        <ThrowingComponent shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(screen.getByText(/Something went wrong/)).toBeInTheDocument();
    expect(screen.getByText('Retry')).toBeInTheDocument();
  });
});
