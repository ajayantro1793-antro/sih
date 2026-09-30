export default function DistrictCard({ name, probability, threshold, alert }) {
  return (
    <div
      className={[
        "relative bg-panel border rounded-md px-4 py-3.5 flex flex-col gap-2",
        alert ? "border-alert/60" : "border-border",
      ].join(" ")}
    >
      <div className="flex items-center justify-between">
        <span className="text-sm text-text">{name}</span>
        <span
          className={[
            "w-2 h-2 rounded-full",
            alert ? "bg-alert animate-pulse_dot" : "bg-normal",
          ].join(" ")}
        />
      </div>

      <div className="font-mono text-2xl font-medium leading-none text-text">
        {(probability * 100).toFixed(1)}
        <span className="text-base text-muted">%</span>
      </div>

      <div className="flex items-center justify-between text-xs text-muted">
        <span className="font-mono">thr {(threshold * 100).toFixed(1)}%</span>
        <span className={alert ? "text-alert" : "text-muted"}>
          {alert ? "Elevated risk" : "Normal"}
        </span>
      </div>
    </div>
  );
}
