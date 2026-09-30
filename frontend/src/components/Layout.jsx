import React from "react";
import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";

export default function Layout() {
  return (
    <div className="min-h-screen w-full bg-[#070b14] flex flex-col md:flex-row" style={{ fontFamily: "'Inter', 'Space Grotesk', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
        .font-mono { font-family: 'JetBrains Mono', ui-monospace, monospace; }
        .font-display { font-family: 'Space Grotesk', sans-serif; }
        @keyframes pulseDot { 0%,100%{opacity:1; transform:scale(1);} 50%{opacity:.5; transform:scale(1.25);} }
        .pulse-dot { animation: pulseDot 1.8s ease-in-out infinite; }
        ::-webkit-scrollbar{width:6px;height:6px;}
        ::-webkit-scrollbar-thumb{background:#243149;border-radius:8px;}
        ::-webkit-scrollbar-track{background:transparent;}
        .bg-scan { background-image: linear-gradient(180deg, rgba(56,189,248,0.04) 1px, transparent 1px); background-size: 100% 3px; }
      `}</style>
      <Sidebar />
      <main className="flex-1 min-w-0">
        <Outlet />
      </main>
    </div>
  );
}
