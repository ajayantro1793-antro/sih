import React from "react";
import { Target } from "lucide-react";

function Stat({ label, value }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/40 px-2.5 py-2 text-center">
      <div className="font-mono text-base font-bold text-slate-100 tabular-nums">{value}</div>
      <div className="text-[10px] text-slate-500">{label}</div>
    </div>
  );
}

/**
 * metrics: metadata.per_district_metrics[districtName], the real,
 * held-out test-set metrics your training script computed -- never
 * recomputed or estimated in the frontend.
 */
export default function PerformanceCard({ district, metrics }) {
  return (
    <div className="rounded-xl border border-slate-700/60 bg-slate-900/60 p-5">
      <div className="flex items-center gap-2 text-slate-300 text-sm font-medium mb-3">
        <Target size={16} className="text-cyan-400" />
        Held-out test performance — {district}
      </div>
      {!metrics ? (
        <div className="text-xs text-slate-600 font-mono py-4 text-center">Loading model metadata…</div>
      ) : (
        <div className="grid grid-cols-3 gap-2">
          <Stat label="AUC-ROC" value={metrics.auc_roc != null ? metrics.auc_roc.toFixed(2) : "—"} />
          <Stat label="Precision" value={metrics.precision != null ? metrics.precision.toFixed(2) : "—"} />
          <Stat label="Recall" value={metrics.recall != null ? metrics.recall.toFixed(2) : "—"} />
          <Stat label="F1" value={metrics.f1 != null ? metrics.f1.toFixed(2) : "—"} />
          <Stat label="Accuracy" value={metrics.accuracy != null ? metrics.accuracy.toFixed(2) : "—"} />
          <Stat label="Threshold" value={metrics.threshold != null ? metrics.threshold.toFixed(2) : "—"} />
        </div>
      )}
      <div className="mt-2.5 text-[10px] text-slate-600 leading-relaxed">
        From a chronological, leakage-safe train/val/test split — this district's test
        period was never seen during training or threshold tuning.
      </div>
    </div>
  );
}
