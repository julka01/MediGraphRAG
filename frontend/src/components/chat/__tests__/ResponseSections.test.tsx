import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { ResponseSections, SourcesSection } from '../ResponseSections';

describe('ResponseSections', () => {
  it('renders fallback when no sections', () => {
    const sections = {
      recommendation: '',
      reasoning: '',
      evidence: '',
      nextSteps: '',
      fallback: 'Fallback content',
    };
    render(<ResponseSections sections={sections} />);
    expect(screen.getByText('Fallback content')).toBeInTheDocument();
  });

  it('renders section titles when content exists', () => {
    const sections = {
      recommendation: 'Take aspirin',
      reasoning: '',
      evidence: 'Study shows...',
      nextSteps: '',
      fallback: '',
    };
    render(<ResponseSections sections={sections} />);
    // Section titles are always rendered as buttons when content exists
    expect(screen.getByText('Summary')).toBeInTheDocument();
    expect(screen.getByText('Evidence')).toBeInTheDocument();
  });

  it('renders source chip when provided', () => {
    const sections = {
      recommendation: 'test',
      reasoning: '',
      evidence: '',
      nextSteps: '',
      fallback: '',
    };
    render(<ResponseSections sections={sections} sourceChip="3 sources · 85% confidence" />);
    expect(screen.getByText(/3 sources/)).toBeInTheDocument();
  });
});

describe('SourcesSection', () => {
  it('renders nothing when no sources', () => {
    const { container } = render(<SourcesSection />);
    expect(container.innerHTML).toBe('');
  });

  it('renders Sources button with edges', () => {
    const edges = [
      { from: '1', to: '2', from_name: 'Aspirin', to_name: 'Pain', relationship: 'TREATS' },
    ];
    render(<SourcesSection reasoningEdges={edges} />);
    expect(screen.getByText('Sources')).toBeInTheDocument();
  });

  it('expands to show edge details when Sources clicked', async () => {
    const edges = [
      { from: '1', to: '2', from_name: 'Aspirin', to_name: 'Pain', relationship: 'TREATS' },
    ];
    render(<SourcesSection reasoningEdges={edges} />);
    await userEvent.click(screen.getByText('Sources'));
    expect(screen.getByText('Aspirin')).toBeInTheDocument();
    expect(screen.getByText('Pain')).toBeInTheDocument();
  });
});
