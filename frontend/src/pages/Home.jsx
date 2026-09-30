import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Satellite, Zap, ArrowRight, Database, Cpu, Target, Map as MapIcon } from "lucide-react";
import { fetchMetadata, fetchConfig } from "../api";

const PIPELINE_STEPS = [
  { icon: Database, title: "Live fetch", desc: "Open-Meteo hourly forecast data across a 23×17 point grid over Tamil Nadu (391 points, 850/500 hPa)." },
  { icon: Cpu, title: "Feature engineering", desc: "IWV, wind shear, lifted-index proxy, 850 hPa humidity, convergence, IWV trend — plus terrain (elevation, slope, TWI)." },
  { icon: Target, title: "Multi-district CNN+LSTM", desc: "One shared backbone, masked per-district pooling — a single model outputs one probability per district, not one model per district." },
  { icon: MapIcon, title: "Per-district thresholds", desc: "Each district gets its own tuned alert threshold from validation-set precision/recall — a coastal district and a dry interior district don't share one cutoff." },
];

function StatBlock({ label, value, sub }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2.5">
      <div className="text-[10px] text-slate-500">{label}</div>
      <div className="font-mono text-lg font-bold text-slate-100 tabular-nums">{value}</div>
      {sub && <div className="text-[10px] text-slate-600">{sub}</div>}
    </div>
  );
}

function averageAuc(metadata) {
  if (!metadata?.per_district_metrics) return null;
  const vals = Object.values(metadata.per_district_metrics)
    .map((m) => m.auc_roc)
    .filter((v) => typeof v === "number");
  if (!vals.length) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

export default function Home() {
  const [meta, setMeta] = useState(null);
  const [config, setConfig] = useState(null);

  useEffect(() => {
    fetchMetadata("thunderstorm").then(setMeta).catch(() => {});
    fetchConfig().then(setConfig).catch(() => {});
  }, []);

  return (
    <div className="min-h-screen text-slate-200 bg-scan">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-10">
        <header className="flex items-center gap-3 mb-2">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500/20 to-cyan-500/5 border border-cyan-500/30 flex items-center justify-center">
            <Satellite size={22} className="text-cyan-400" />
          </div>
          <div>
            <h1 className="font-display text-2xl font-bold text-white">Weather Nowcasting AI</h1>
            <p className="text-sm text-slate-500">Tamil Nadu — Thunderstorm Nowcasting</p>
          </div>
        </header>
        <p className="text-sm text-slate-500 max-w-2xl mb-8">
          Per-district thunderstorm risk from live Open-Meteo forecast data, using a multi-district
          CNN+LSTM model trained on two years of historical data across 13 Tamil Nadu districts.
        </p>

        {/* Hero CTA */}
        <Link
          to="/thunderstorm-tn"
          className="block rounded-2xl border border-amber-500/30 bg-gradient-to-br from-amber-500/10 to-transparent p-8 mb-10 hover:border-amber-500/50 transition-colors group"
        >
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-xl bg-amber-500/15 border border-amber-500/30 flex items-center justify-center">
                <Zap size={26} className="text-amber-400" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-display text-lg font-bold text-white">Thunderstorm Dashboard</span>
                  <span className="flex items-center gap-1 text-[10px] font-bold text-emerald-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 pulse-dot" /> LIVE
                  </span>
                </div>
                <div className="text-sm text-slate-400 mt-0.5">
                  {meta ? `Avg. test AUC ${(averageAuc(meta) * 100 || 0).toFixed(0)}% across ${meta.districts?.length ?? "—"} districts` : "Loading model stats…"}
                </div>
              </div>
            </div>
            <ArrowRight size={22} className="text-slate-500 group-hover:text-amber-400 group-hover:translate-x-1 transition-all" />
          </div>
        </Link>

        {/* Pipeline */}
        <section className="mb-10">
          <h2 className="font-display text-sm font-semibold text-slate-300 mb-3">How a prediction is made</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {PIPELINE_STEPS.map((s, i) => (
              <div key={i} className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
                <div className="flex items-center gap-2 mb-2">
                  <s.icon size={15} className="text-cyan-400" />
                  <span className="text-[10px] font-mono text-slate-600">STEP {i + 1}</span>
                </div>
                <div className="text-sm font-medium text-slate-200 mb-1">{s.title}</div>
                <div className="text-xs text-slate-500 leading-relaxed">{s.desc}</div>
              </div>
            ))}
          </div>
        </section>

        {/* Real training stats */}
        <section className="mb-10">
          <h2 className="font-display text-sm font-semibold text-slate-300 mb-3">Real training statistics</h2>
          <div className="rounded-xl border border-slate-700/60 bg-slate-900/60 p-5">
            {!meta ? (
              <div className="text-xs text-slate-600 font-mono">loading…</div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                <StatBlock label="Train samples" value={meta.train_samples} />
                <StatBlock label="Val samples" value={meta.val_samples} />
                <StatBlock label="Test samples" value={meta.test_samples} />
                <StatBlock label="Features" value={meta.features?.length ?? "—"} />
                <StatBlock label="Districts" value={meta.districts?.length ?? "—"} />
                <StatBlock label="Avg test AUC" value={averageAuc(meta) ? averageAuc(meta).toFixed(2) : "—"} />
              </div>
            )}
          </div>
        </section>

        {/* System parameters */}
        {config && (
          <section>
            <h2 className="font-display text-sm font-semibold text-slate-300 mb-3">System parameters</h2>
            <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 flex flex-wrap gap-x-8 gap-y-2 text-xs font-mono text-slate-500">
              <span>Region: <span className="text-slate-300">{config.region_name}</span></span>
              <span>Lead time: <span className="text-slate-300">{config.lead_time_hours}h ahead</span></span>
              <span>Sequence: <span className="text-slate-300">{config.sequence_timesteps} hourly timesteps</span></span>
              <span>Districts: <span className="text-slate-300">{config.n_districts}</span></span>
              <span>Bounds: <span className="text-slate-300">{config.lat_min}–{config.lat_max}°N, {config.lon_min}–{config.lon_max}°E</span></span>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
