import { render, screen } from '@testing-library/react';
import { AppProvider } from '../../../context/AppContext';
import { MainLayout } from '../MainLayout';

function Wrapper({ children }: { children: React.ReactNode }) {
  return <AppProvider>{children}</AppProvider>;
}

describe('MainLayout', () => {
  const graphPanel = <div data-testid="graph">Graph</div>;
  const chatPanel = <div data-testid="chat">Chat</div>;

  it('renders both panels in split layout', () => {
    render(<MainLayout layout="split" graphPanel={graphPanel} chatPanel={chatPanel} />, { wrapper: Wrapper });
    expect(screen.getByTestId('graph')).toBeInTheDocument();
    expect(screen.getByTestId('chat')).toBeInTheDocument();
  });

  it('renders only graph in graph-only layout', () => {
    render(<MainLayout layout="graph-only" graphPanel={graphPanel} chatPanel={chatPanel} />, { wrapper: Wrapper });
    expect(screen.getByTestId('graph')).toBeInTheDocument();
    expect(screen.queryByTestId('chat')).not.toBeInTheDocument();
  });

  it('renders only chat in chat-only layout', () => {
    render(<MainLayout layout="chat-only" graphPanel={graphPanel} chatPanel={chatPanel} />, { wrapper: Wrapper });
    expect(screen.queryByTestId('graph')).not.toBeInTheDocument();
    expect(screen.getByTestId('chat')).toBeInTheDocument();
  });
});
