import React from "react";
import { NavLink } from "react-router-dom";
import { Satellite, Home as HomeIcon, Zap } from "lucide-react";

// PRESENTATION BUILD: only Overview + Thunderstorm are linked. To restore
// Cloudburst, add back:
//   { to: "/cloudburst-tn", label: "Cloudburst — TN", icon: CloudRain }
const NAV_ITEMS = [
  { to: "/", label: "Overview", icon: HomeIcon, end: true },
  { to: "/thunderstorm-tn", label: "Thunderstorm — TN", icon: Zap },
];

export default function Sidebar() {
  return (
    <aside
      className="md:w-60 md:h-screen md:sticky md:top-0 flex-shrink-0 border-b md:border-b-0 md:border-r border-slate-800/80 bg-slate-950/70 backdrop-blur"
      style={{ fontFamily: "'Inter', sans-serif" }}
    >
      <div className="flex items-center gap-2.5 px-4 py-4 md:py-5 border-b border-slate-800/60">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500/20 to-cyan-500/5 border border-cyan-500/30 flex items-center justify-center flex-shrink-0">
          <Satellite size={18} className="text-cyan-400" />
        </div>
        <div className="min-w-0">
          <div className="font-display font-bold text-sm text-white leading-tight truncate">Weather Nowcasting AI</div>
          <div className="text-[10px] text-slate-500 font-mono">Tamil Nadu</div>
        </div>
      </div>

      <nav className="flex md:flex-col gap-1 p-2 overflow-x-auto md:overflow-visible">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              `flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-sm whitespace-nowrap transition-colors ${
                isActive
                  ? "bg-cyan-500/10 text-cyan-300 border border-cyan-500/30"
                  : "text-slate-400 border border-transparent hover:bg-slate-900 hover:text-slate-200"
              }`
            }
          >
            <item.icon size={16} />
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
