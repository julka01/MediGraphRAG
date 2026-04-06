import { useEffect } from 'react';
import { useApp } from '../../context/AppContext';

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
    <div className="toast toast-end toast-bottom z-50">
      <div className={`alert ${alertClass} shadow-lg`}>
        <span>{notif.message}</span>
        <button
          className="btn btn-ghost btn-xs"
          onClick={() => dispatch({ type: 'CLEAR_NOTIFICATION' })}
        >
          ✕
        </button>
      </div>
    </div>
  );
}

export function showError(dispatch, message) {
  dispatch({ type: 'SHOW_NOTIFICATION', notifType: 'error', message });
}

export function showSuccess(dispatch, message) {
  dispatch({ type: 'SHOW_NOTIFICATION', notifType: 'success', message });
}
