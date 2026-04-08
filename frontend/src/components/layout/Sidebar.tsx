import { MoonIcon, SunIcon } from '@heroicons/react/24/outline';
import clsx from 'clsx';
import type { ReactNode } from 'react';
import { useApp } from '../../context/AppContext';
import { useTheme } from '../../context/ThemeContext';
import type { HealthResponse } from '../../types/app';
import { HealthDot } from '../ui/HealthDot';

interface SidebarProps {
  children: ReactNode;
  width?: number;
}

function SidebarRoot({ children, width }: SidebarProps) {
  const { state } = useApp();

  const style = state.panels.leftCollapsed ? { width: 0, minWidth: 0 } : width ? { width } : undefined;
  const className = clsx(
    'flex flex-col bg-base-200 transition-all duration-300 overflow-y-auto',
    state.panels.leftCollapsed ? 'overflow-hidden' : width ? 'shrink-0' : 'w-72 min-w-72',
  );

  return (
    <div className={className} style={style}>
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
        {theme === 'dark' ? (
          <MoonIcon className="size-5" aria-hidden="true" />
        ) : (
          <SunIcon className="size-5" aria-hidden="true" />
        )}
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
