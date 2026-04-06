import type { ReactNode } from 'react';
import { AppProvider } from './AppContext';
import { GraphProvider } from './GraphContext';
import { KGProvider } from './KGContext';
import { ThemeProvider } from './ThemeContext';
import { UIProvider } from './UIContext';

export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider>
      <UIProvider>
        <KGProvider>
          <GraphProvider>
            <AppProvider>{children}</AppProvider>
          </GraphProvider>
        </KGProvider>
      </UIProvider>
    </ThemeProvider>
  );
}
