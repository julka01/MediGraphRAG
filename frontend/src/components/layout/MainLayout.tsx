import type { ReactNode } from 'react';
import type { Layout } from '../../types/app';

interface MainLayoutProps {
  layout: Layout;
  graphPanel: ReactNode;
  chatPanel: ReactNode;
}

export function MainLayout({ layout, graphPanel, chatPanel }: MainLayoutProps) {
  const showGraph = layout !== 'chat-only';
  const showChat = layout !== 'graph-only';
  const graphFlex = layout === 'split' ? 'flex-[1.6]' : 'flex-1';

  return (
    <div className="flex flex-col md:flex-row flex-1 min-w-0">
      {showGraph && <div className={`border-b md:border-b-0 md:border-r border-base-300 overflow-hidden ${graphFlex}`}>{graphPanel}</div>}
      {showChat && <div className="overflow-hidden flex-1">{chatPanel}</div>}
    </div>
  );
}
