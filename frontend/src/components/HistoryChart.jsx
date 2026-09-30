import React, { useMemo } from "react";
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";

/**
 * rows: raw log rows from /api/log/{hazard} (all districts, all logged runs).
 * selectedDistrict: district name to filter+plot.
 */
export default function HistoryChart({ rows, selectedDistrict }) {
  const data = useMemo(() => {
    if (!rows) return [];
    return rows
      .filter((r) => r.district === selectedDistrict)
      .map((r) => ({
        label: new Date(r.logged_at_utc).toLocaleString("en-IN", {
          hour12: false, day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
        }),
        probability: Number(r.probability) * 100,
        threshold: Number(r.threshold) * 100,
      }));
  }, [rows, selectedDistrict]);

  return (
    <div className="rounded-xl border border-slate-700/60 bg-slate-900/60 p-4">
      <div className="flex items-center justify-between mb-1">
        <h2 className="font-display text-sm font-semibold text-slate-300">Live Prediction History — {selectedDistrict}</h2>
        <span className="text-[10px] text-slate-600 font-mono">real logged runs · not a future forecast</span>
      </div>
      <div className="h-64 mt-2">
        {data.length === 0 ? (
          <div className="h-full flex items-center justify-center text-xs text-slate-600 font-mono">
            No logged history yet for this district — fetch live data a few times to build a track record.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 6, right: 10, left: -18, bottom: 0 }}>
              <defs>
                <linearGradient id="gProb" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#38bdf8" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="label" stroke="#475569" tick={{ fontSize: 9, fontFamily: "JetBrains Mono" }} interval="preserveStartEnd" />
              <YAxis stroke="#475569" tick={{ fontSize: 10, fontFamily: "JetBrains Mono" }} domain={[0, 100]} width={30} />
              <RTooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 11, fontFamily: "JetBrains Mono" }}
                labelStyle={{ color: "#94a3b8" }}
                formatter={(v, name) => [`${v.toFixed(1)}%`, name === "probability" ? "Probability" : "Threshold"]}
              />
              <Area type="monotone" dataKey="probability" stroke="none" fill="url(#gProb)" fillOpacity={1} isAnimationActive={false} />
              <Line type="monotone" dataKey="probability" stroke="#38bdf8" strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
              <Line type="monotone" dataKey="threshold" stroke="#64748b" strokeWidth={1} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>
      <div className="flex items-center gap-4 mt-1 text-[11px] font-mono">
        <span className="flex items-center gap-1.5 text-sky-400"><span className="w-2.5 h-0.5 bg-sky-400 inline-block" />Probability</span>
        <span className="flex items-center gap-1.5 text-slate-500"><span className="w-2.5 h-0.5 bg-slate-500 inline-block" style={{ borderTop: "1px dashed" }} />Tuned threshold</span>
      </div>
    </div>
  );
}
