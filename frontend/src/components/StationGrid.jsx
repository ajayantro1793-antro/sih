import React from "react";
import { MapPin } from "lucide-react";
import { riskLevel } from "../riskLevel";

export default function StationGrid({ districts, selectedName, onSelect }) {
  const sorted = [...districts].sort((a, b) => b.probability - a.probability);

  return (
    <section className="mt-5 rounded-xl border border-slate-700/60 bg-slate-900/60 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <MapPin size={15} className="text-cyan-400" />
          <h2 className="font-display text-sm font-semibold text-slate-300 tracking-wide">Tamil Nadu — Statewide Risk Overview</h2>
        </div>
        <span className="text-[10px] text-slate-600 font-mono">tap a district to inspect · sorted by risk</span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 lg:grid-cols-7 gap-2">
        {sorted.map((d) => {
          const pct = d.probability * 100;
          const rl = riskLevel(pct);
          const active = d.name === selectedName;
          return (
            <button
              key={d.name}
              onClick={() => onSelect(d.name)}
              className={`text-left rounded-lg border p-2.5 transition-colors ${active ? "bg-slate-800/80" : "bg-slate-950/40 hover:bg-slate-800/50"}`}
              style={{ borderColor: active ? rl.color : `${rl.color}33` }}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-200 truncate">{d.name}</span>
                <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: rl.color }} />
              </div>
              <div className="text-[10px] text-slate-500 font-mono mb-1">{d.alert ? "alert" : "normal"}</div>
              <div className="font-mono text-base font-bold tabular-nums" style={{ color: rl.color }}>
                {pct.toFixed(0)}%
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}
