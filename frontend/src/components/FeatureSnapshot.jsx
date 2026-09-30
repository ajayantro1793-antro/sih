import React from "react";
import { Layers } from "lucide-react";
import { featureLabel } from "../featureLabels";

export default function FeatureSnapshot({ features, exclude = [] }) {
  const entries = features
    ? Object.entries(features).filter(([name]) => !exclude.includes(name))
    : [];

  return (
    <div className="rounded-xl border border-slate-700/60 bg-slate-900/60 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Layers size={15} className="text-cyan-400" />
          <h2 className="font-display text-sm font-semibold text-slate-300">Live Feature Snapshot</h2>
        </div>
        <span className="text-[10px] text-slate-600 font-mono">area-mean · this model's actual inputs</span>
      </div>
      {entries.length === 0 ? (
        <div className="text-xs text-slate-600 font-mono py-6 text-center">No live data fetched yet.</div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {entries.map(([name, value]) => {
            const { label, unit } = featureLabel(name);
            return (
              <div key={name} className="rounded-lg border border-slate-800 bg-slate-950/40 px-2.5 py-2">
                <div className="text-[10px] text-slate-500 truncate">{label}</div>
                <div className="font-mono text-sm text-slate-200 tabular-nums">
                  {value.toFixed(Math.abs(value) >= 100 ? 0 : 3)}
                  <span className="text-slate-600 text-[10px] ml-1">{unit}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
      <div className="mt-3 text-[10px] text-slate-600 leading-relaxed">
        Not a SHAP or feature-importance analysis — these are simply the model's real current inputs
        (region-averaged, most recent timestep), no explainability has been computed for this MVP.
      </div>
    </div>
  );
}
