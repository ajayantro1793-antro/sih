import React from "react";
import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import HazardDashboard from "./pages/HazardDashboard";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Home />} />
        <Route path="/thunderstorm-tn" element={<HazardDashboard hazard="thunderstorm" />} />
        <Route path="/cloudburst-tn" element={<HazardDashboard hazard="cloudburst" />} />
      </Route>
    </Routes>
  );
}
