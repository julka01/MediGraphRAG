import { render, screen } from '@testing-library/react';
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

  it('renders sections without source chip', () => {
    const sections = {
      recommendation: 'test',
      reasoning: '',
      evidence: '',
      nextSteps: '',
      fallback: '',
    };
    render(<ResponseSections sections={sections} />);
    expect(screen.getByText('Summary')).toBeInTheDocument();
  });
});

describe('SourcesSection', () => {
  it('renders nothing when no sources', () => {
    const { container } = render(<SourcesSection />);
    expect(container.innerHTML).toBe('');
  });

  it('renders source header and edges inline', () => {
    const edges = [{ from: '1', to: '2', from_name: 'Aspirin', to_name: 'Pain', relationship: 'TREATS' }];
    render(<SourcesSection reasoningEdges={edges} />);
    expect(screen.getByText('1 Source')).toBeInTheDocument();
    expect(screen.getByText('Aspirin')).toBeInTheDocument();
    expect(screen.getByText('Pain')).toBeInTheDocument();
  });
});
