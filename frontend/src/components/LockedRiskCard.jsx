import React from "react";
import { Lock } from "lucide-react";

export default function LockedRiskCard({ title, icon: Icon }) {
  return (
    <div className="rounded-xl border border-slate-800 p-5 relative overflow-hidden bg-slate-900/30 opacity-60">
      <div className="flex items-center justify-between relative">
        <div className="flex items-center gap-2 text-slate-500 text-sm font-medium">
          <Icon size={16} />
          {title}
        </div>
        <span className="text-[10px] font-bold tracking-wide px-2 py-0.5 rounded-full bg-slate-800 text-slate-500 flex items-center gap-1">
          <Lock size={9} /> NOT BUILT
        </span>
      </div>
      <div className="mt-3 text-sm text-slate-600 leading-relaxed">
        No dataset, labels, or trained model yet for this hazard.
      </div>
    </div>
  );
}
