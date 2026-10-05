import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Zap, CloudRain, Droplets, RefreshCw, Clock, Wifi, WifiOff, AlertTriangle, ArrowRight } from "lucide-react";
import RiskCard from "../components/RiskCard";
import PerformanceCard from "../components/PerformanceCard";
import LockedRiskCard from "../components/LockedRiskCard";
import StationGrid from "../components/StationGrid";
import Gauge from "../components/Gauge";
import FeatureSnapshot from "../components/FeatureSnapshot";
import HistoryChart from "../components/HistoryChart";
import AlertFeed from "../components/AlertFeed";
import { fetchConfig, fetchMetadata } from "../api";
import { featureLabel } from "../featureLabels";
import { useHazardData } from "../store/PredictionProvider";

const HAZARD_META = {
  thunderstorm: {
    title: "Thunderstorm",
    icon: Zap,
    color: "#fbbf24",
    otherHazard: { to: "/cloudburst-tn", label: "Cloudburst", icon: CloudRain },
    caveat:
      "Uses Open-Meteo operational forecast data — the same source this model was trained on — plus rectangular district-box approximations (src/districts.py). Proof of concept only, not a validated operational forecast.",
  },
  cloudburst: {
    title: "Cloudburst",
    icon: CloudRain,
    color: "#38bdf8",
    otherHazard: { to: "/thunderstorm-tn", label: "Thunderstorm", icon: Zap },
    caveat:
      "Cloudburst labels are an extreme-rainfall proxy (top ~3% of forward rainfall windows per district), not a confirmed cloudburst event record — true cloudbursts are sub-grid-scale, sub-hourly events this hourly/0.25° pipeline cannot resolve directly. Includes terrain (DEM) features: elevation, slope, and a simplified Topographic Wetness Index.",
  },
};

function useClock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now;
}

function formatRefreshInterval(ms) {
  const min = Math.round(ms / 60000);
  return `${min} min`;
}

export default function HazardDashboard({ hazard }) {
  const meta = HAZARD_META[hazard];
  const now = useClock();

  const {
    result, prevDistrictsByName, selectedDistrict, logRows,
    loading, error, lastFetchedAt, refetch, setSelectedDistrict, autoRefreshMs,
  } = useHazardData(hazard);

  const [config, setConfig] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [backendUp, setBackendUp] = useState(null);

  useEffect(() => {
    fetchConfig().then(setConfig).catch(() => {});
    fetchMetadata(hazard).then(setMetadata).catch(() => {});
    fetch((import.meta.env.VITE_API_BASE || "http://localhost:8000") + "/api/health")
      .then((r) => setBackendUp(r.ok))
      .catch(() => setBackendUp(false));
  }, [hazard]);

  const selectedEntry = result?.districts.find((d) => d.name === selectedDistrict);
  const trend =
    selectedEntry && prevDistrictsByName && prevDistrictsByName[selectedEntry.name] !== undefined
      ? (selectedEntry.probability - prevDistrictsByName[selectedEntry.name]) * 100
      : null;

  const iwv = result?.features?.iwv;
  const windShear = result?.features?.wind_shear;
  const districtMetrics = metadata?.per_district_metrics?.[selectedDistrict];

  return (
    <div className="min-h-screen text-slate-200 bg-scan">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        {/* Header */}
        <header className="flex flex-wrap items-start justify-between gap-4 mb-6">
          <div className="flex items-center gap-3">
            <div
              className="w-11 h-11 rounded-xl flex items-center justify-center border"
              style={{ background: `${meta.color}1a`, borderColor: `${meta.color}44` }}
            >
              <meta.icon size={20} style={{ color: meta.color }} />
            </div>
            <div>
              <h1 className="font-display text-xl font-bold text-white leading-tight">{meta.title} — Tamil Nadu</h1>
              <div className="flex items-center gap-3 text-[11px] text-slate-500 font-mono mt-0.5">
                <span className="flex items-center gap-1"><Clock size={11} />{now.toLocaleTimeString("en-IN", { hour12: false })} IST</span>
                <span className="flex items-center gap-1">
                  {backendUp ? <Wifi size={11} className="text-emerald-400" /> : <WifiOff size={11} className="text-red-400" />}
                  {backendUp === null ? "checking..." : backendUp ? "backend online" : "backend unreachable"}
                </span>
              </div>
            </div>
          </div>

          <div className="flex flex-col items-end gap-1">
            <button
              onClick={refetch}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium bg-white text-slate-950 hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
            >
              <RefreshCw size={15} className={loading ? "animate-spin" : ""} />
              {loading ? "Fetching live data…" : "Refresh now"}
            </button>
            <span className="text-[10px] text-slate-600 font-mono">
              {result?.cache_age_seconds != null
                ? `server data age: ${Math.round(result.cache_age_seconds / 60)} min`
                : `auto-refreshes every ${formatRefreshInterval(autoRefreshMs)}`}
            </span>
          </div>
        </header>

        <p className="text-xs text-slate-500 leading-relaxed mb-5 max-w-3xl">{meta.caveat}</p>

        {error && !result && (
          <div className="mb-5 rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-300 flex items-center gap-2">
            <AlertTriangle size={15} /> {error}
          </div>
        )}

        {result && (
          <div className="mb-3 text-[11px] text-slate-500 font-mono">
            Based on live atmospheric data through{" "}
            <span className="text-slate-300">{result.latest_time}</span>
            {result.cache_age_seconds != null && (
              <span className="ml-2 text-slate-600">
                (server updated {Math.round(result.cache_age_seconds / 60)} min ago)
              </span>
            )}
          </div>
        )}

        {!result && loading && (
          <div className="mb-5 rounded-lg border border-slate-800 bg-slate-900/40 px-4 py-6 text-center text-sm text-slate-500 font-mono">
            Loading prediction data — server is warming up, please wait ~60 s…
          </div>
        )}

        {/* Risk card + real performance + other hazard + locked Flash Flood */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {selectedEntry ? (
            <RiskCard
              title={`${meta.title} — ${selectedEntry.name}`}
              icon={meta.icon}
              pct={selectedEntry.probability * 100}
              trend={trend}
              subtitle={`threshold ${(selectedEntry.threshold * 100).toFixed(1)}% (tuned per-district) · updated ${
                lastFetchedAt ? new Date(lastFetchedAt).toLocaleTimeString("en-IN", { hour12: false }) : ""
              }`}
            />
          ) : (
            <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 flex items-center justify-center text-xs text-slate-600 font-mono">
              Waiting for first live fetch…
            </div>
          )}

          <PerformanceCard district={selectedDistrict || "—"} metrics={districtMetrics} />

          <Link
            to={meta.otherHazard.to}
            className="rounded-xl border border-slate-700/60 bg-slate-900/60 p-5 flex flex-col justify-between hover:border-slate-500 transition-colors group"
          >
            <div className="flex items-center gap-2 text-slate-300 text-sm font-medium">
              <meta.otherHazard.icon size={16} />
              {meta.otherHazard.label} model
            </div>
            <div className="flex items-center justify-between mt-3">
              <span className="text-xs text-slate-500">Also live — switch to check it</span>
              <ArrowRight size={16} className="text-slate-600 group-hover:text-slate-300 group-hover:translate-x-0.5 transition-all" />
            </div>
          </Link>

          <LockedRiskCard title="Flash Flood" icon={Droplets} />
        </div>

        {/* Statewide overview */}
        {result && (
          <StationGrid
            districts={result.districts}
            selectedName={selectedDistrict}
            onSelect={setSelectedDistrict}
          />
        )}

        {/* Gauges + feature snapshot */}
        {result && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-5">
            <div className="grid grid-cols-2 gap-4 lg:col-span-1">
              {iwv !== undefined && (
                <Gauge
                  label={featureLabel("iwv").label}
                  value={iwv}
                  unit={featureLabel("iwv").unit}
                  min={0}
                  max={80}
                  icon={CloudRain}
                  zones={[
                    { upTo: 30, label: "Dry", color: "#34d399" },
                    { upTo: 50, label: "Moderate", color: "#fbbf24" },
                    { upTo: 65, label: "Moist", color: "#fb923c" },
                    { upTo: 999, label: "Saturated", color: "#ef4444" },
                  ]}
                />
              )}
              {windShear !== undefined && (
                <Gauge
                  label={featureLabel("wind_shear").label}
                  value={windShear}
                  unit={featureLabel("wind_shear").unit}
                  min={0}
                  max={25}
                  icon={Zap}
                  zones={[
                    { upTo: 8, label: "Weak", color: "#34d399" },
                    { upTo: 14, label: "Moderate", color: "#fbbf24" },
                    { upTo: 20, label: "Strong", color: "#fb923c" },
                    { upTo: 999, label: "Severe", color: "#ef4444" },
                  ]}
                />
              )}
            </div>
            <div className="lg:col-span-2">
              <FeatureSnapshot features={result.features} exclude={["iwv", "wind_shear"]} />
            </div>
          </div>
        )}

        {/* History + alerts */}
        {result && (
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 mt-5">
            <div className="xl:col-span-2">
              <HistoryChart rows={logRows} selectedDistrict={selectedDistrict} />
            </div>
            <AlertFeed districts={result.districts} latestTime={result.latest_time} hazardLabel={meta.title.toLowerCase()} />
          </div>
        )}

        {/* Real system parameters */}
        {config && (
          <div className="mt-5 rounded-xl border border-slate-800 bg-slate-900/40 p-4 flex flex-wrap gap-x-8 gap-y-2 text-xs font-mono text-slate-500">
            <span>Region: <span className="text-slate-300">{config.region_name}</span></span>
            <span>Lead time: <span className="text-slate-300">{config.lead_time_hours}h</span></span>
            <span>Sequence: <span className="text-slate-300">{config.sequence_timesteps} timesteps</span></span>
            <span>Districts: <span className="text-slate-300">{config.n_districts}</span></span>
            <span>Bounds: <span className="text-slate-300">{config.lat_min}–{config.lat_max}°N, {config.lon_min}–{config.lon_max}°E</span></span>
          </div>
        )}
      </div>
    </div>
  );
}
