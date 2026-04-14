import { XMarkIcon } from '@heroicons/react/24/outline';
import clsx from 'clsx';
import { useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import type { AppAction } from '../../types/app';

export function Notifications() {
  const { state, dispatch } = useApp();
  const notif = state.notification;

  useEffect(() => {
    if (!notif) return;
    if (notif.type === 'success') {
      const timer = setTimeout(() => dispatch({ type: 'CLEAR_NOTIFICATION' }), 5000);
      return () => clearTimeout(timer);
    }
  }, [notif, dispatch]);

  if (!notif) return null;

  const isError = notif.type === 'error';

  return (
    <div className="toast toast-top toast-end z-50" role="status" aria-live="polite">
      <div className={clsx('alert max-w-xs shadow-lg', isError ? 'alert-error' : 'alert-success')}>
        <span className="flex-1 text-sm">{notif.message}</span>
        <button
          type="button"
          className="btn btn-ghost btn-xs btn-circle opacity-60 hover:opacity-100"
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
