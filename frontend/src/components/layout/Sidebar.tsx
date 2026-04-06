import type { ReactNode } from 'react';
import { useApp } from '../../context/AppContext';
import { useTheme } from '../../context/ThemeContext';
import type { HealthResponse } from '../../types/app';
import { HealthDot } from '../ui/HealthDot';

interface SidebarProps {
  children: ReactNode;
}

function SidebarRoot({ children }: SidebarProps) {
  const { state, dispatch } = useApp();

  return (
    <div
      className={`relative flex flex-col bg-base-200 border-r border-base-300 transition-all duration-300 overflow-y-auto ${
        state.sidebarCollapsed ? 'w-0 min-w-0 overflow-hidden' : 'w-72 min-w-72'
      }`}
    >
      <button
        type="button"
        className="absolute top-2 -right-6 z-30 btn btn-ghost btn-xs"
        onClick={() => dispatch({ type: 'TOGGLE_SIDEBAR' })}
        title={state.sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        aria-label={state.sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {state.sidebarCollapsed ? '›' : '‹'}
      </button>

      <div className="p-3 space-y-4">{children}</div>
    </div>
  );
}

interface SidebarHeaderProps {
  healthData?: HealthResponse | null;
}

function SidebarHeader({ healthData }: SidebarHeaderProps) {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="flex items-center justify-between">
      <button type="button" className="btn btn-ghost btn-sm" onClick={toggleTheme} aria-label="Toggle theme">
        {theme === 'dark' ? '🌙' : '☀️'}
      </button>
      <HealthDot initialData={healthData} />
    </div>
  );
}

interface SidebarSectionProps {
  title: string;
  children: ReactNode;
}

function SidebarSection({ title, children }: SidebarSectionProps) {
  return (
    <div>
      <h3 className="font-semibold text-sm mb-2">{title}</h3>
      {children}
    </div>
  );
}

export const Sidebar = Object.assign(SidebarRoot, {
  Header: SidebarHeader,
  Section: SidebarSection,
});
