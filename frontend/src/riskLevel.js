export function riskLevel(pct) {
  if (pct >= 70) return { label: "Severe", color: "#ef4444", glow: "rgba(239,68,68,0.35)" };
  if (pct >= 45) return { label: "High", color: "#fb923c", glow: "rgba(251,146,60,0.35)" };
  if (pct >= 20) return { label: "Moderate", color: "#fbbf24", glow: "rgba(251,191,36,0.3)" };
  return { label: "Low", color: "#34d399", glow: "rgba(52,211,153,0.3)" };
}
