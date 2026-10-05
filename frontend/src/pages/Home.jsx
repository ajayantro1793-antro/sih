import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Satellite, Zap, CloudRain, Droplets, ArrowRight, Lock, Database,
  Cpu, Target, Map as MapIcon, CheckCircle2,
} from "lucide-react";
import { fetchMetadata, fetchConfig } from "../api";

const PIPELINE_STEPS = [
  { icon: Database, title: "Live fetch", desc: "Open-Meteo hourly forecast data across a 23×17 point grid over Tamil Nadu (391 points, 850/500 hPa)." },
  { icon: Cpu, title: "Feature engineering", desc: "IWV, wind shear, lifted-index proxy, 850 hPa humidity, convergence, IWV trend — plus terrain (elevation, slope, TWI) and, for cloudburst, moisture-flux convergence and 500 hPa humidity." },
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
  const [tMeta, setTMeta] = useState(null);
  const [cMeta, setCMeta] = useState(null);
  const [config, setConfig] = useState(null);

  useEffect(() => {
    fetchMetadata("thunderstorm").then(setTMeta).catch(() => {});
    fetchMetadata("cloudburst").then(setCMeta).catch(() => {});
    fetchConfig().then(setConfig).catch(() => {});
  }, []);

  const models = [
    {
      key: "thunderstorm", title: "Thunderstorm", icon: Zap, color: "#fbbf24",
      to: "/thunderstorm-tn", live: true, meta: tMeta,
      desc: "Heavy-rainfall proxy (85th percentile of forward rainfall), per district.",
    },
    {
      key: "cloudburst", title: "Cloudburst", icon: CloudRain, color: "#38bdf8",
      to: "/cloudburst-tn", live: true, meta: cMeta,
      desc: "Extreme-rainfall proxy (97th percentile) + terrain (DEM) features.",
    },
    {
      key: "flashflood", title: "Flash Flood", icon: Droplets, color: "#64748b",
      to: null, live: false, meta: null,
      desc: "Not built — no dataset, labels, or trained model yet.",
    },
  ];

  return (
    <div className="min-h-screen text-slate-200 bg-scan">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        <header className="flex items-center gap-3 mb-2">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500/20 to-cyan-500/5 border border-cyan-500/30 flex items-center justify-center">
            <Satellite size={22} className="text-cyan-400" />
          </div>
          <div>
            <h1 className="font-display text-2xl font-bold text-white">Weather Nowcasting AI</h1>
            <p className="text-sm text-slate-500">Tamil Nadu — Smart India Hackathon 2026</p>
          </div>
        </header>
        <p className="text-sm text-slate-500 max-w-2xl mb-8">
          Per-district severe-weather nowcasting from live Open-Meteo forecast data. Two hazards are
          trained and live right now; a third is on the roadmap.
        </p>

        <section className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-10">
          {models.map((m) => {
            const CardInner = (
              <div
                className={`rounded-xl border p-5 h-full flex flex-col transition-colors ${
                  m.live ? "border-slate-700/60 bg-slate-900/60 hover:border-slate-500" : "border-slate-800 bg-slate-900/30 opacity-60"
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: `${m.color}1a` }}>
                    <m.icon size={17} style={{ color: m.color }} />
                  </div>
                  {m.live ? (
                    <span className="flex items-center gap-1 text-[10px] font-bold text-emerald-400">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 pulse-dot" /> LIVE
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-[10px] font-bold text-slate-600">
                      <Lock size={10} /> ROADMAP
                    </span>
                  )}
                </div>
                <div className="font-display font-semibold text-white">{m.title}</div>
                <div className="text-xs text-slate-500 mt-1 flex-1">{m.desc}</div>
                {m.live && (
                  <div className="flex items-center justify-between mt-4 text-xs">
                    <span className="text-slate-500 font-mono">
                      {m.meta ? `AUC ${(averageAuc(m.meta) * 100 || 0).toFixed(0)}% avg` : "loading…"}
                    </span>
                    <ArrowRight size={14} className="text-slate-600" />
                  </div>
                )}
              </div>
            );
            return m.to ? (
              <Link key={m.key} to={m.to}>{CardInner}</Link>
            ) : (
              <div key={m.key}>{CardInner}</div>
            );
          })}
        </section>

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

        <section className="mb-10">
          <h2 className="font-display text-sm font-semibold text-slate-300 mb-3">Real training statistics</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {[{ label: "Thunderstorm", meta: tMeta, color: "#fbbf24" }, { label: "Cloudburst", meta: cMeta, color: "#38bdf8" }].map(
              ({ label, meta, color }) => (
                <div key={label} className="rounded-xl border border-slate-700/60 bg-slate-900/60 p-4">
                  <div className="text-sm font-medium mb-3" style={{ color }}>{label}</div>
                  {!meta ? (
                    <div className="text-xs text-slate-600 font-mono">loading…</div>
                  ) : (
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                      <StatBlock label="Train samples" value={meta.train_samples} />
                      <StatBlock label="Val samples" value={meta.val_samples} />
                      <StatBlock label="Test samples" value={meta.test_samples} />
                      <StatBlock label="Features" value={meta.features?.length ?? "—"} />
                      <StatBlock label="Districts" value={meta.districts?.length ?? "—"} />
                      <StatBlock label="Avg test AUC" value={averageAuc(meta) ? averageAuc(meta).toFixed(2) : "—"} />
                    </div>
                  )}
                </div>
              )
            )}
          </div>
        </section>

        {config && (
          <section className="mb-10">
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

        <section>
          <h2 className="font-display text-sm font-semibold text-slate-300 mb-3">Roadmap</h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              { done: true, text: "Thunderstorm model — trained, live" },
              { done: true, text: "Cloudburst model + DEM/terrain features — trained, live" },
              { done: false, text: "Flash flood model — needs its own label definition, dataset, training" },
            ].map((r, i) => (
              <div key={i} className="rounded-xl border border-slate-800 bg-slate-900/40 p-3.5 flex items-start gap-2.5">
                {r.done ? (
                  <CheckCircle2 size={16} className="text-emerald-400 flex-shrink-0 mt-0.5" />
                ) : (
                  <Lock size={16} className="text-slate-600 flex-shrink-0 mt-0.5" />
                )}
                <span className={`text-xs leading-relaxed ${r.done ? "text-slate-300" : "text-slate-500"}`}>{r.text}</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
