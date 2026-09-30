const OPTIONS = [
  { key: "thunderstorm", label: "Thunderstorm" },
  { key: "cloudburst", label: "Cloudburst" },
];

export default function HazardToggle({ value, onChange }) {
  return (
    <div className="inline-flex rounded-md border border-border bg-panelmuted p-1">
      {OPTIONS.map((opt) => {
        const active = opt.key === value;
        return (
          <button
            key={opt.key}
            onClick={() => onChange(opt.key)}
            className={[
              "px-4 py-1.5 text-sm rounded transition-colors duration-150",
              active
                ? "bg-panel text-text border border-border"
                : "text-muted hover:text-text border border-transparent",
            ].join(" ")}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
