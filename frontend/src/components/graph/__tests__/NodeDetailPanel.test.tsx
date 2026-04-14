import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { NodeDetailPanel } from '../NodeDetailPanel';

describe('NodeDetailPanel', () => {
  const mockNode = {
    label: 'Aspirin',
    originalId: '42',
    labels: ['Drug'],
    properties: { dosage: '100mg', route: 'oral' },
  };

  it('renders node label and type', () => {
    render(<NodeDetailPanel node={mockNode} nodeColor="#ff0000" edges={[]} onClose={vi.fn()} />);
    expect(screen.getByText('Aspirin')).toBeInTheDocument();
    expect(screen.getByText('Drug')).toBeInTheDocument();
  });

  it('renders properties', () => {
    render(<NodeDetailPanel node={mockNode} nodeColor="#ff0000" edges={[]} onClose={vi.fn()} />);
    expect(screen.getByText('dosage')).toBeInTheDocument();
    expect(screen.getByText('100mg')).toBeInTheDocument();
    expect(screen.getByText('route')).toBeInTheDocument();
    expect(screen.getByText('oral')).toBeInTheDocument();
  });

  it('calls onClose when close button clicked', async () => {
    const onClose = vi.fn();
    render(<NodeDetailPanel node={mockNode} nodeColor="#ff0000" edges={[]} onClose={onClose} />);
    await userEvent.click(screen.getByLabelText('Close node details'));
    expect(onClose).toHaveBeenCalled();
  });

  it('shows empty message when no properties', () => {
    const emptyNode = { ...mockNode, properties: {} };
    render(<NodeDetailPanel node={emptyNode} nodeColor="#ff0000" edges={[]} onClose={vi.fn()} />);
    expect(screen.getByText('No additional properties')).toBeInTheDocument();
  });
});
