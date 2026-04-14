import type { ReactNode } from 'react';

interface SidebarProps {
  children: ReactNode;
  hidden?: boolean;
}

export function Sidebar({ children, hidden }: SidebarProps) {
  return (
    <aside className={`relative flex flex-col bg-base-200 w-72 min-w-72 shrink-0 overflow-y-auto${hidden ? ' hidden' : ''}`}>
      <div className="flex flex-col gap-3 p-3 min-w-0">
        {children}
      </div>
    </aside>
  );
}
