export function MetricCard({
  label,
  value,
  sub,
  accent = false,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
}) {
  return (
    <div className="metric">
      <div className="metric__label">{label}</div>
      <div className={`metric__value${accent ? " metric__value--accent" : ""}`}>{value}</div>
      {sub ? <div className="metric__sub">{sub}</div> : null}
    </div>
  );
}
