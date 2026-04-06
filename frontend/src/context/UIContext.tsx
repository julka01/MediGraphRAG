import { createContext, useContext, useReducer } from 'react';
import type { Layout, Notification, NotificationType } from '../types/app';

interface UIState {
  layout: Layout;
  sidebarCollapsed: boolean;
  notification: Notification | null;
}

type UIAction =
  | { type: 'SET_LAYOUT'; layout: Layout }
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'SHOW_NOTIFICATION'; notifType: NotificationType; message: string }
  | { type: 'CLEAR_NOTIFICATION' };

interface UIContextValue {
  state: UIState;
  dispatch: React.Dispatch<UIAction>;
}

const UIContext = createContext<UIContextValue | null>(null);

const initialState: UIState = {
  layout: 'split',
  sidebarCollapsed: false,
  notification: null,
};

function uiReducer(state: UIState, action: UIAction): UIState {
  switch (action.type) {
    case 'SET_LAYOUT':
      return { ...state, layout: action.layout };
    case 'TOGGLE_SIDEBAR':
      return { ...state, sidebarCollapsed: !state.sidebarCollapsed };
    case 'SHOW_NOTIFICATION':
      return { ...state, notification: { type: action.notifType, message: action.message } };
    case 'CLEAR_NOTIFICATION':
      return { ...state, notification: null };
    default:
      return state;
  }
}

export function UIProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(uiReducer, initialState);
  return <UIContext.Provider value={{ state, dispatch }}>{children}</UIContext.Provider>;
}

export function useUI(): UIContextValue {
  const ctx = useContext(UIContext);
  if (!ctx) throw new Error('useUI must be used within UIProvider');
  return ctx;
}

export type { UIAction, UIState };
