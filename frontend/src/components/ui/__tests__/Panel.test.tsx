import { render, screen } from '@testing-library/react';
import { Panel } from '../Panel';

describe('Panel', () => {
  it('renders header, body, and footer', () => {
    render(
      <Panel>
        <Panel.Header title="Test Panel" />
        <Panel.Body>
          <p>Content here</p>
        </Panel.Body>
        <Panel.Footer>
          <button type="button">Action</button>
        </Panel.Footer>
      </Panel>,
    );

    expect(screen.getByText('Test Panel')).toBeInTheDocument();
    expect(screen.getByText('Content here')).toBeInTheDocument();
    expect(screen.getByText('Action')).toBeInTheDocument();
  });

  it('renders header with badge and actions', () => {
    render(
      <Panel>
        <Panel.Header title="Graph" badge={<span data-testid="badge">KG1</span>}>
          <button type="button">Export</button>
        </Panel.Header>
        <Panel.Body>Body</Panel.Body>
      </Panel>,
    );

    expect(screen.getByText('Graph')).toBeInTheDocument();
    expect(screen.getByTestId('badge')).toBeInTheDocument();
    expect(screen.getByText('Export')).toBeInTheDocument();
  });

  it('renders body as scrollable by default', () => {
    render(
      <Panel>
        <Panel.Body>Scrollable</Panel.Body>
      </Panel>,
    );

    const body = screen.getByText('Scrollable').closest('[data-panel-body]');
    expect(body?.className).toContain('overflow-y-auto');
  });
});
