import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

// Minimal mock of useApp to drive the Notifications component.
const mockDispatch = vi.fn();

vi.mock('../../../context/AppContext', () => ({
  useApp: () => ({
    state: {
      notification: { type: 'error', message: 'Something broke' },
    },
    dispatch: mockDispatch,
  }),
}));

// Import AFTER mock is registered.
const { Notifications } = await import('../Notifications');

describe('Notifications', () => {
  it('renders an alert inside a toast container', () => {
    const { container } = render(<Notifications />);
    expect(container.querySelector('.toast')).not.toBeNull();
    expect(container.querySelector('.alert')).not.toBeNull();
    expect(screen.getByText('Something broke')).toBeInTheDocument();
  });

  it('uses alert-error class for error notifications', () => {
    const { container } = render(<Notifications />);
    const alert = container.querySelector('.alert');
    expect(alert?.className).toContain('alert-error');
  });

  it('dismisses when X button is clicked', async () => {
    render(<Notifications />);
    await userEvent.click(screen.getByLabelText('Dismiss notification'));
    expect(mockDispatch).toHaveBeenCalledWith({ type: 'CLEAR_NOTIFICATION' });
  });
});
