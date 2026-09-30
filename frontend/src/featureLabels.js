export const FEATURE_LABELS = {
  iwv: { label: "IWV (moisture proxy)", unit: "mm" },
  wind_shear: { label: "Wind Shear (850–500 hPa)", unit: "m/s" },
  lifted_index_proxy: { label: "Lifted Index Proxy", unit: "K" },
  humidity_850: { label: "Humidity @ 850 hPa", unit: "kg/kg" },
  convergence: { label: "Low-Level Convergence", unit: "s⁻¹" },
  iwv_trend: { label: "IWV Trend", unit: "mm/step" },
  moisture_flux_convergence: { label: "Moisture Flux Convergence", unit: "proxy" },
  humidity_500: { label: "Humidity @ 500 hPa", unit: "kg/kg" },
  elevation_m: { label: "Elevation (terrain)", unit: "m" },
  slope_degrees: { label: "Slope (terrain)", unit: "°" },
  twi: { label: "Topographic Wetness Index", unit: "index" },
};

export function featureLabel(name) {
  return FEATURE_LABELS[name] || { label: name, unit: "" };
}
