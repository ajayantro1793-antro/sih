import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from "react";
import { fetchLivePrediction, fetchLog } from "../api";

// How often each hazard auto-refreshes itself, in milliseconds. The
// underlying Open-Meteo data is hourly, so refreshing much faster than
// this mostly just re-confirms the same numbers -- 10 minutes keeps the
// dashboard feeling "live" without hammering the live-fetch pipeline
// (each call does a full 391-point Open-Meteo fetch + model rebuild).
// Change this one constant to adjust the interval for both hazards.
const AUTO_REFRESH_MS = 10 * 60 * 1000;

const HAZARDS = ["thunderstorm", "cloudburst"];

const PredictionContext = createContext(null);

function emptyHazardState() {
  return {
    result: null,
    prevDistrictsByName: null,
    selectedDistrict: null,
    logRows: null,
    loading: false,
    error: null,
    lastFetchedAt: null,
  };
}

/**
 * Wraps the whole app (above the router) so hazard prediction state
 * survives navigation between /thunderstorm-tn and /cloudburst-tn --
 * those pages read from here instead of owning their own local state,
 * so switching tabs never resets what's already been fetched.
 *
 * Also owns the auto-refresh timers for both hazards, started once here
 * rather than per-page, so a hazard keeps refreshing in the background
 * even while you're looking at the OTHER hazard's page.
 */
export function PredictionProvider({ children }) {
  const [state, setState] = useState(() => ({
    thunderstorm: emptyHazardState(),
    cloudburst: emptyHazardState(),
  }));

  // Mirrors `state` for use inside callbacks without becoming a dependency
  // (avoids recreating fetchHazard / re-running the interval-setup effect
  // every time state changes).
  const stateRef = useRef(state);
  stateRef.current = state;

  const patchHazard = useCallback((hazard, patch) => {
    setState((prev) => ({
      ...prev,
      [hazard]: { ...prev[hazard], ...(typeof patch === "function" ? patch(prev[hazard]) : patch) },
    }));
  }, []);

  const fetchHazard = useCallback(
    async (hazard) => {
      if (stateRef.current[hazard].loading) return; // don't overlap two fetches for the same hazard
      patchHazard(hazard, { loading: true, error: null });
      try {
        const data = await fetchLivePrediction(hazard);
        const prevResult = stateRef.current[hazard].result;
        let prevDistrictsByName = stateRef.current[hazard].prevDistrictsByName;
        if (prevResult) {
          const byName = {};
          prevResult.districts.forEach((d) => (byName[d.name] = d.probability));
          prevDistrictsByName = byName;
        }
        patchHazard(hazard, (prevHazardState) => ({
          result: data,
          prevDistrictsByName,
          selectedDistrict:
            prevHazardState.selectedDistrict ||
            [...data.districts].sort((a, b) => b.probability - a.probability)[0]?.name ||
            null,
          loading: false,
          error: null,
          lastFetchedAt: Date.now(),
        }));
        fetchLog(hazard)
          .then((d) => patchHazard(hazard, { logRows: d.rows }))
          .catch(() => {});
      } catch (err) {
        patchHazard(hazard, (prevHazardState) => ({
          loading: false,
          error: prevHazardState.result ? null : (err.message || "Live fetch failed."),
          result: prevHazardState.result,
        }));
      }
    },
    [patchHazard]
  );

  const setSelectedDistrict = useCallback(
    (hazard, name) => patchHazard(hazard, { selectedDistrict: name }),
    [patchHazard]
  );

  // Fetch hazards on app load (staggered to prevent hitting Open-Meteo simultaneously),
  // then keep refreshing on a timer.
  const initialFetchStarted = useRef(false);

  useEffect(() => {
    if (!initialFetchStarted.current) {
      initialFetchStarted.current = true;
      HAZARDS.forEach((hazard, idx) => {
        setTimeout(() => {
          fetchHazard(hazard);
          fetchLog(hazard)
            .then((d) => patchHazard(hazard, { logRows: d.rows }))
            .catch(() => {});
        }, idx * 4000);
      });
    }

    const interval = setInterval(() => {
      HAZARDS.forEach((hazard, idx) => {
        setTimeout(() => fetchHazard(hazard), idx * 4000);
      });
    }, AUTO_REFRESH_MS);

    return () => clearInterval(interval);
    // Intentionally empty deps -- this should run exactly once per app load.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <PredictionContext.Provider value={{ state, fetchHazard, setSelectedDistrict, autoRefreshMs: AUTO_REFRESH_MS }}>
      {children}
    </PredictionContext.Provider>
  );
}

/** Per-hazard view into the shared store, used by HazardDashboard. */
export function useHazardData(hazard) {
  const ctx = useContext(PredictionContext);
  if (!ctx) throw new Error("useHazardData must be used within a PredictionProvider");
  return {
    ...ctx.state[hazard],
    refetch: () => ctx.fetchHazard(hazard),
    setSelectedDistrict: (name) => ctx.setSelectedDistrict(hazard, name),
    autoRefreshMs: ctx.autoRefreshMs,
  };
}
