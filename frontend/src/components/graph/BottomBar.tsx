import { FilterChips } from './FilterChips';
import { GraphControls } from './GraphControls';

interface BottomBarProps {
  height: number;
}

export function BottomBar({ height }: BottomBarProps) {
  return (
    <div className="flex flex-col border-t border-base-content/10 bg-base-200/60 backdrop-blur-xl" style={{ height }}>
      {/* Layer 1: Fixed controls */}
      <div className="shrink-0 border-b border-base-content/10">
        <GraphControls />
      </div>

      {/* Layer 2: Scrollable filter chips */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        <FilterChips />
      </div>
    </div>
  );
}
