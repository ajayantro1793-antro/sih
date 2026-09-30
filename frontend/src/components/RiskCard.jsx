import React from "react";
import { riskLevel } from "../riskLevel";

/**
 * trend is in PERCENTAGE POINTS vs. the previous live fetch for this same
 * district, or null if there's no previous fetch to compare against yet
 * (first fetch of the session) -- never fabricated.
 */
export default function RiskCard({ title, icon: Icon, pct, trend, subtitle }) {
  const rl = riskLevel(pct);
  return (
    <div
      className="rounded-xl border p-5 relative overflow-hidden bg-slate-900/60"
      style={{ borderColor: `${rl.color}44`, boxShadow: `0 0 0 1px rgba(0,0,0,0), 0 8px 24px -12px ${rl.glow}` }}
    >
      <div className="absolute -right-6 -top-6 w-24 h-24 rounded-full blur-2xl opacity-30" style={{ background: rl.color }} />
      <div className="flex items-center justify-between relative">
        <div className="flex items-center gap-2 text-slate-300 text-sm font-medium">
          <Icon size={16} style={{ color: rl.color }} />
          {title}
        </div>
        <span
          className="text-[10px] font-bold tracking-wide px-2 py-0.5 rounded-full"
          style={{ background: `${rl.color}22`, color: rl.color, border: `1px solid ${rl.color}55` }}
        >
          {rl.label.toUpperCase()}
        </span>
      </div>
      <div className="mt-3 flex items-end gap-2 relative">
        <span className="text-4xl font-bold font-mono tabular-nums" style={{ color: rl.color }}>
          {pct.toFixed(1)}%
        </span>
        {trend !== null && (
          <span
            className={`mb-1.5 text-xs font-mono flex items-center gap-0.5 ${
              trend > 0.05 ? "text-red-400" : trend < -0.05 ? "text-emerald-400" : "text-slate-500"
            }`}
          >
            {trend > 0.05 ? "▲" : trend < -0.05 ? "▼" : "—"} {Math.abs(trend).toFixed(1)}%
          </span>
        )}
      </div>
      <div className="mt-3 h-1.5 rounded-full bg-slate-800 overflow-hidden relative">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, background: rl.color }} />
      </div>
      <div className="mt-1.5 text-[10px] text-slate-500 font-mono">{subtitle}</div>
    </div>
  );
}
