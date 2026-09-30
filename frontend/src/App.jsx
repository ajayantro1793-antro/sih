import React from "react";
import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import HazardDashboard from "./pages/HazardDashboard";

// PRESENTATION BUILD (Dr. Kalam Young Achiever Awards 2026): thunderstorm
// only. The cloudburst model/backend are untouched and fully working --
// to bring the cloudburst route back after the presentation, restore the
// "/cloudburst-tn" <Route> here and the corresponding nav item in
// Sidebar.jsx (both are just commented out below, not deleted elsewhere).
export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Home />} />
        <Route path="/thunderstorm-tn" element={<HazardDashboard hazard="thunderstorm" />} />
        {/* <Route path="/cloudburst-tn" element={<HazardDashboard hazard="cloudburst" />} /> */}
      </Route>
    </Routes>
  );
}
