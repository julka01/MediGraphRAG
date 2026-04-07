import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { AppProvider } from '../../../context/AppContext';
import { GraphFilters } from '../GraphFilters';

// Use a real AppProvider so we don't have to mock the context module.
// GraphFilters reads nodeTypeColors / relationshipTypeColors from state,
// which start as empty objects — so we only verify the Filters button
// and the dispatch actions via button clicks.
function Wrapper({ children }: { children: React.ReactNode }) {
  return <AppProvider>{children}</AppProvider>;
}

describe('GraphFilters', () => {
  it('renders Filters button', () => {
    render(<GraphFilters />, { wrapper: Wrapper });
    expect(screen.getByText('Filters')).toBeInTheDocument();
  });

  // DaisyUI dropdown is CSS-only — dropdown-content is always in the DOM.
  it('renders Apply and Reset buttons in dropdown content', () => {
    render(<GraphFilters />, { wrapper: Wrapper });
    expect(screen.getByText('Apply')).toBeInTheDocument();
    expect(screen.getByText('Reset')).toBeInTheDocument();
  });

  it('renders Graph Filters heading in dropdown', () => {
    render(<GraphFilters />, { wrapper: Wrapper });
    expect(screen.getByText('Graph Filters')).toBeInTheDocument();
  });

  // With an empty initial state, no node/rel type checkboxes are shown.
  // Click Apply — should call dispatch with SET_FILTERS action (no-op with empty sets).
  it('Apply button is clickable without throwing', () => {
    render(<GraphFilters />, { wrapper: Wrapper });
    expect(() => fireEvent.click(screen.getByText('Apply'))).not.toThrow();
  });

  it('Reset button is clickable without throwing', () => {
    render(<GraphFilters />, { wrapper: Wrapper });
    expect(() => fireEvent.click(screen.getByText('Reset'))).not.toThrow();
  });
});
