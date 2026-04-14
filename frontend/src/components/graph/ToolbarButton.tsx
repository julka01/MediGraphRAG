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
        'btn btn-xs rounded-xl border border-base-content/12 bg-base-100/60 shadow-none transition-all hover:-translate-y-px hover:bg-base-100/85',
        active ? 'border-primary/25 text-primary shadow-sm' : 'text-base-content/60 hover:text-base-content/82',
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}
