import type { ReactNode } from 'react';
import { AppProvider } from './AppContext';
import { ThemeProvider } from './ThemeContext';

export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider>
      <AppProvider>{children}</AppProvider>
    </ThemeProvider>
  );
}
