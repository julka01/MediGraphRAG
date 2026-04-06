import type { ReactNode } from 'react';

interface PanelProps {
  children: ReactNode;
}

function PanelRoot({ children }: PanelProps) {
  return <div className="flex flex-col h-full">{children}</div>;
}

interface PanelHeaderProps {
  title: string;
  badge?: ReactNode;
  children?: ReactNode;
}

function PanelHeader({ title, badge, children }: PanelHeaderProps) {
  return (
    <div className="flex items-center justify-between px-2 py-1">
      <div className="flex items-center gap-2">
        <h2 className="text-lg font-bold">{title}</h2>
        {badge}
      </div>
      {children && <div className="flex gap-1">{children}</div>}
    </div>
  );
}

interface PanelBodyProps {
  children: ReactNode;
  scrollable?: boolean;
}

function PanelBody({ children, scrollable = true }: PanelBodyProps) {
  return (
    <div data-panel-body="" className={`flex-1 min-h-0 px-2 ${scrollable ? 'overflow-y-auto' : 'overflow-hidden'}`}>
      {children}
    </div>
  );
}

interface PanelFooterProps {
  children: ReactNode;
}

function PanelFooter({ children }: PanelFooterProps) {
  return <div className="px-2 py-1 border-t border-base-300">{children}</div>;
}

export const Panel = Object.assign(PanelRoot, {
  Header: PanelHeader,
  Body: PanelBody,
  Footer: PanelFooter,
});
