import { memo } from 'react';

interface PanelToggleIconProps {
  panel: 'left' | 'bottom' | 'right' | 'top';
  isOpen: boolean;
  onClick: () => void;
}

function LeftPanelSvg({ filled }: { filled: boolean }) {
  return (
    <svg width="22" height="22" viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <rect x="2" y="2" width="44" height="44" rx="5" stroke="currentColor" strokeWidth="2.5" fill="none" />
      {filled ? (
        <>
          <rect x="3" y="3" width="12" height="42" rx="4" fill="currentColor" />
          <rect x="9" y="3" width="6" height="42" fill="currentColor" />
        </>
      ) : null}
      <line x1="15" y1="6" x2="15" y2="42" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function TopPanelSvg({ filled }: { filled: boolean }) {
  return (
    <svg width="22" height="22" viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <rect x="2" y="2" width="44" height="44" rx="5" stroke="currentColor" strokeWidth="2.5" fill="none" />
      {filled ? (
        <>
          <rect x="3" y="3" width="42" height="12" fill="currentColor" />
          <rect x="3" y="3" width="42" height="6" rx="4" fill="currentColor" />
        </>
      ) : null}
      <line x1="6" y1="15" x2="42" y2="15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function BottomPanelSvg({ filled }: { filled: boolean }) {
  return (
    <svg width="22" height="22" viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <rect x="2" y="2" width="44" height="44" rx="5" stroke="currentColor" strokeWidth="2.5" fill="none" />
      {filled ? (
        <>
          <rect x="3" y="33" width="42" height="12" fill="currentColor" />
          <rect x="3" y="39" width="42" height="6" rx="4" fill="currentColor" />
        </>
      ) : null}
      <line x1="6" y1="33" x2="42" y2="33" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function RightPanelSvg({ filled }: { filled: boolean }) {
  return (
    <svg width="22" height="22" viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <rect x="2" y="2" width="44" height="44" rx="5" stroke="currentColor" strokeWidth="2.5" fill="none" />
      {filled ? (
        <>
          <rect x="33" y="3" width="12" height="42" rx="4" fill="currentColor" />
          <rect x="33" y="3" width="6" height="42" fill="currentColor" />
        </>
      ) : null}
      <line x1="33" y1="6" x2="33" y2="42" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

const svgMap = { left: LeftPanelSvg, bottom: BottomPanelSvg, right: RightPanelSvg, top: TopPanelSvg };
const labelMap = { left: 'Toggle left panel', bottom: 'Toggle bottom panel', right: 'Toggle right panel', top: 'Toggle top panel' };

export const PanelToggleIcon = memo(function PanelToggleIcon({ panel, isOpen, onClick }: PanelToggleIconProps) {
  const SvgComponent = svgMap[panel];
  return (
    <button
      type="button"
      onClick={onClick}
      className={`p-1 rounded-md transition-colors ${isOpen ? 'text-primary' : 'text-base-content/50 hover:text-base-content/80'}`}
      aria-label={labelMap[panel]}
      aria-pressed={isOpen}
    >
      <SvgComponent filled={isOpen} />
    </button>
  );
});
