import clsx from 'clsx';
import type { ButtonHTMLAttributes } from 'react';

interface ToolbarButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  active?: boolean;
}

export function ToolbarButton({ active, className, children, ...props }: ToolbarButtonProps) {
  return (
    <button
      type="button"
      className={clsx(
        'btn btn-xs shadow-none border border-base-content/20 hover:bg-base-content/5 transition-colors bg-transparent',
        active ? 'text-brand' : 'text-base-content/60 hover:text-base-content/80',
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}
