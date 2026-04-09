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
    <div className="fixed top-3 right-3 z-50" role="status" aria-live="polite">
      <div
        className={clsx(
          'flex items-center gap-2 bg-base-100 border rounded-lg px-3 py-2 shadow-lg max-w-xs',
          isError ? 'border-error/40 border-l-3 border-l-error' : 'border-success/40 border-l-3 border-l-success',
        )}
      >
        <span className="flex-1 text-sm text-base-content">{notif.message}</span>
        <button
          type="button"
          className="opacity-50 hover:opacity-100 transition-opacity shrink-0"
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
