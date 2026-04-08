import type { ReactNode } from 'react';
import { AppProvider } from './AppContext';
import { GraphProvider } from './GraphContext';
import { KGProvider } from './KGContext';
import { ThemeProvider } from './ThemeContext';
export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider>
      <KGProvider>
        <GraphProvider>
          <AppProvider>{children}</AppProvider>
        </GraphProvider>
      </KGProvider>
    </ThemeProvider>
  );
}
