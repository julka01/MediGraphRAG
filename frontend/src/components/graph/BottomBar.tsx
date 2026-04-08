import { FilterChips } from './FilterChips';
import { GraphControls } from './GraphControls';

interface BottomBarProps {
  height: number;
}

export function BottomBar({ height }: BottomBarProps) {
  return (
    <div className="bg-base-200 flex flex-col" style={{ height }}>
      {/* Layer 1: Fixed controls */}
      <div className="shrink-0 border-b border-base-300/30">
        <GraphControls />
      </div>

      {/* Layer 2: Scrollable filter chips */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        <FilterChips />
      </div>
    </div>
  );
}
