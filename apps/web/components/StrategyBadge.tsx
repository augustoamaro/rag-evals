export function StrategyBadge({ strategy }: { strategy: string }) {
  return <span className={`badge badge--${strategy}`}>{strategy}</span>;
}
