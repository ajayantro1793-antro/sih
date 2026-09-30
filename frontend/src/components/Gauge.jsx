import React from "react";
import { ResponsiveContainer, RadialBarChart, RadialBar, PolarAngleAxis } from "recharts";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export default function Gauge({ label, value, unit, min, max, zones, invert, icon: Icon }) {
  const pct = invert
    ? clamp(((max - value) / (max - min)) * 100, 0, 100)
    : clamp(((value - min) / (max - min)) * 100, 0, 100);

  const zoneAt = zones.find((z) => (invert ? value <= z.upTo : value <= z.upTo)) || zones[zones.length - 1];
  const color = zoneAt.color;

  const data = [{ name: label, value: pct, fill: color }];

  return (
    <div className="relative rounded-xl border border-slate-700/60 bg-slate-900/60 p-4 flex flex-col items-center">
      <div className="flex items-center gap-1.5 self-start text-[11px] tracking-wide text-slate-400 font-mono mb-1">
        <Icon size={13} className="text-cyan-400" />
        {label}
      </div>
      <div className="w-full h-[104px] -mt-1">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            innerRadius="72%"
            outerRadius="100%"
            data={data}
            startAngle={180}
            endAngle={0}
            barSize={11}
          >
            <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
            <RadialBar background={{ fill: "#1e293b" }} dataKey="value" cornerRadius={6} fill={color} />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      <div className="-mt-9 flex flex-col items-center">
        <span className="text-2xl font-bold font-mono tabular-nums" style={{ color }}>
          {value.toFixed(Math.abs(value) >= 100 ? 0 : 2)}
        </span>
        <span className="text-[10px] text-slate-500 -mt-0.5">{unit}</span>
      </div>
      <div className="mt-2 px-2 py-0.5 rounded-full text-[10px] font-semibold tracking-wide"
        style={{ background: `${color}22`, color }}>
        {zoneAt.label}
      </div>
      <div className="flex gap-1 mt-2">
        {zones.map((z, i) => (
          <span key={i} className="w-4 h-1 rounded-full" style={{ background: z.color, opacity: zoneAt.label === z.label ? 1 : 0.25 }} />
        ))}
      </div>
    </div>
  );
}
