import { Cog6ToothIcon } from '@heroicons/react/24/outline';
import type { ReactNode } from 'react';

interface SidebarProps {
  children: ReactNode;
  onSettingsClick: () => void;
}

export function Sidebar({ children, onSettingsClick }: SidebarProps) {
  return (
    <aside className="relative flex flex-col bg-base-200 w-72 min-w-72 shrink-0 overflow-y-auto">
      <div className="flex flex-col gap-3 p-3 pb-14">
        {children}
      </div>
      <button
        type="button"
        onClick={onSettingsClick}
        className="absolute bottom-3 left-3 z-10 size-7 rounded-full bg-base-300 flex items-center justify-center shadow-md hover:bg-base-300/80 transition-colors"
        aria-label="Database Settings"
        title="Database Settings"
      >
        <Cog6ToothIcon className="size-4 text-base-content/60" />
      </button>
    </aside>
  );
}
