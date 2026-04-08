import { render, screen } from '@testing-library/react';
import { AppProvider } from '../../../context/AppContext';
import { MainLayout } from '../MainLayout';

function Wrapper({ children }: { children: React.ReactNode }) {
  return <AppProvider>{children}</AppProvider>;
}

describe('MainLayout', () => {
  const graphPanel = <div data-testid="graph">Graph</div>;
  const chatPanel = <div data-testid="chat">Chat</div>;
  const bottomBar = <div data-testid="bottom">Bottom</div>;

  it('renders graph, chat, and bottom bar panels', () => {
    render(<MainLayout graphPanel={graphPanel} chatPanel={chatPanel} bottomBar={bottomBar} />, { wrapper: Wrapper });
    expect(screen.getByTestId('graph')).toBeInTheDocument();
    expect(screen.getByTestId('chat')).toBeInTheDocument();
    expect(screen.getByTestId('bottom')).toBeInTheDocument();
  });

  it('renders graph panel always', () => {
    render(<MainLayout graphPanel={graphPanel} chatPanel={chatPanel} bottomBar={bottomBar} />, { wrapper: Wrapper });
    expect(screen.getByTestId('graph')).toBeInTheDocument();
  });
});
