import { createContext, useContext, useReducer } from 'react';
import type { KGListItem } from '../types/app';

interface KGState {
  currentKGId: string | null;
  currentKGName: string | null;
  kgList: KGListItem[];
}

type KGAction =
  | { type: 'SET_KG'; kgId: string; kgName: string | null }
  | { type: 'SET_KG_LIST'; kgList: KGListItem[] }
  | { type: 'CLEAR_KG' };

interface KGContextValue {
  state: KGState;
  dispatch: React.Dispatch<KGAction>;
}

const KGContext = createContext<KGContextValue | null>(null);

const initialState: KGState = {
  currentKGId: null,
  currentKGName: null,
  kgList: [],
};

function kgReducer(state: KGState, action: KGAction): KGState {
  switch (action.type) {
    case 'SET_KG':
      return { ...state, currentKGId: action.kgId, currentKGName: action.kgName };
    case 'SET_KG_LIST':
      return { ...state, kgList: action.kgList };
    case 'CLEAR_KG':
      return { ...state, currentKGId: null, currentKGName: null };
    default:
      return state;
  }
}

export function KGProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(kgReducer, initialState);
  return <KGContext.Provider value={{ state, dispatch }}>{children}</KGContext.Provider>;
}

export function useKG(): KGContextValue {
  const ctx = useContext(KGContext);
  if (!ctx) throw new Error('useKG must be used within KGProvider');
  return ctx;
}

export type { KGAction, KGState };
