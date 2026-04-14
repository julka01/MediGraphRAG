import type { ReactNode } from 'react';

interface SidebarProps {
  children: ReactNode;
  hidden?: boolean;
}

export function Sidebar({ children, hidden }: SidebarProps) {
  return (
    <aside
      className={`relative flex flex-col w-[16.5rem] min-w-[16.5rem] shrink-0 overflow-y-auto border-r border-base-content/10 bg-base-200/70 backdrop-blur-xl ${
        hidden ? ' hidden' : ''
      }`}
    >
      <div className="flex flex-col gap-4 p-4 min-w-0">{children}</div>
    </aside>
  );
}
