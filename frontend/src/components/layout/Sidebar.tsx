import type { ReactNode } from 'react';

interface SidebarProps {
  children: ReactNode;
}

export function Sidebar({ children }: SidebarProps) {
  return (
    <aside className="flex flex-col bg-base-200 w-72 min-w-72 shrink-0 overflow-y-auto">
      <div className="flex flex-col gap-3 p-3">
        {children}
      </div>
    </aside>
  );
}
