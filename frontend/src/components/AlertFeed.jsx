import React from "react";
import { AlertOctagon, AlertTriangle, Radio, Info } from "lucide-react";

const SEVERITY_STYLE = {
  SEVERE: { color: "#ef4444", icon: AlertOctagon },
  WARNING: { color: "#fb923c", icon: AlertTriangle },
  WATCH: { color: "#fbbf24", icon: Radio },
  INFO: { color: "#38bdf8", icon: Info },
};

function severityFor(margin) {
  // margin = probability - threshold, both 0-1. Purely a display banding
  // over how far above its own tuned threshold a district is -- not a
  // claim about any specific physical mechanism.
  if (margin >= 0.3) return "SEVERE";
  if (margin >= 0.15) return "WARNING";
  return "WATCH";
}

/**
 * districts: the real districts array from the last /api/live/{hazard}
 * response. latestTime: that response's latest_time. hazardLabel: e.g.
 * "thunderstorm" / "cloudburst", used in the message text.
 */
export default function AlertFeed({ districts, latestTime, hazardLabel }) {
  const alerted = districts
    ? [...districts].filter((d) => d.alert).sort((a, b) => (b.probability - b.threshold) - (a.probability - a.threshold))
    : [];

  return (
    <div className="rounded-xl border border-slate-700/60 bg-slate-900/60 flex flex-col min-h-[420px] xl:h-auto">
      <div className="p-4 pb-2 flex items-center justify-between border-b border-slate-800">
        <h2 className="font-display text-sm font-semibold text-slate-300">Alert Feed</h2>
        {districts && (
          <span className="flex items-center gap-1.5 text-[10px] font-mono text-emerald-400">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 pulse-dot" /> LIVE
          </span>
        )}
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-2 max-h-[520px]">
        {!districts && (
          <div className="text-xs text-slate-600 font-mono p-2">No live data fetched yet.</div>
        )}

        {districts && alerted.length === 0 && (
          <div className="rounded-lg border p-2.5" style={{ borderColor: `${SEVERITY_STYLE.INFO.color}33`, background: `${SEVERITY_STYLE.INFO.color}0d` }}>
            <div className="flex items-center justify-between mb-1">
              <span className="flex items-center gap-1.5 text-[10px] font-bold tracking-wide" style={{ color: SEVERITY_STYLE.INFO.color }}>
                <Info size={12} /> INFO
              </span>
            </div>
            <div className="text-xs text-slate-300 leading-snug">
              No district currently shows elevated {hazardLabel} risk.
            </div>
          </div>
        )}

        {alerted.map((d) => {
          const margin = d.probability - d.threshold;
          const sev = severityFor(margin);
          const s = SEVERITY_STYLE[sev];
          const Icon = s.icon;
          return (
            <div key={d.name} className="rounded-lg border p-2.5" style={{ borderColor: `${s.color}33`, background: `${s.color}0d` }}>
              <div className="flex items-center justify-between mb-1">
                <span className="flex items-center gap-1.5 text-[10px] font-bold tracking-wide" style={{ color: s.color }}>
                  <Icon size={12} /> {sev}
                </span>
                <span className="text-[10px] font-mono text-slate-500">{latestTime ? new Date(latestTime).toLocaleTimeString("en-IN", { hour12: false, hour: "2-digit", minute: "2-digit" }) : ""}</span>
              </div>
              <div className="text-[11px] text-slate-400 font-mono mb-0.5">{d.name}</div>
              <div className="text-xs text-slate-300 leading-snug">
                Predicted {hazardLabel} probability {(d.probability * 100).toFixed(1)}% exceeds tuned alert threshold of {(d.threshold * 100).toFixed(1)}%.
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
