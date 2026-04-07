import { useEffect } from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import { useApp } from '../../context/AppContext';
import type { AppAction } from '../../types/app';

export function Notifications() {
  const { state, dispatch } = useApp();
  const notif = state.notification;

  useEffect(() => {
    if (!notif) return;
    if (notif.type === 'success') {
      const timer = setTimeout(() => dispatch({ type: 'CLEAR_NOTIFICATION' }), 3000);
      return () => clearTimeout(timer);
    }
  }, [notif, dispatch]);

  if (!notif) return null;

  const alertClass = notif.type === 'error' ? 'alert-error' : 'alert-success';

  return (
    <div className="toast toast-end toast-bottom z-50" role="status" aria-live="polite">
      <div className={`alert ${alertClass} shadow-lg`}>
        <span>{notif.message}</span>
        <button
          type="button"
          className="btn btn-ghost btn-xs"
          onClick={() => dispatch({ type: 'CLEAR_NOTIFICATION' })}
          aria-label="Dismiss notification"
        >
          <XMarkIcon className="size-4" aria-hidden="true" />
        </button>
      </div>
    </div>
  );
}

export function showError(dispatch: React.Dispatch<AppAction>, message: string): void {
  dispatch({ type: 'SHOW_NOTIFICATION', notifType: 'error', message });
}

export function showSuccess(dispatch: React.Dispatch<AppAction>, message: string): void {
  dispatch({ type: 'SHOW_NOTIFICATION', notifType: 'success', message });
}
