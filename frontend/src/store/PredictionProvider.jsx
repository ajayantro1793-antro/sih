import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from "react";
import { fetchLivePrediction, fetchLog } from "../api";

// Open-Meteo's free tier is rate-limited aggressively, so automatic
// refreshes must be much less frequent than they are in a typical demo.
// Keeping it at 1 hour avoids hammering the API while still making the
// dashboard feel alive for a deployed project.
const AUTO_REFRESH_MS = 60 * 60 * 1000;

// PRESENTATION BUILD (Dr. Kalam Young Achiever Awards 2026): only
// thunderstorm auto-fetches/auto-refreshes, so the demo doesn't spend
// time on a live-fetch cycle for a hazard that isn't being shown. The
// store itself is still hazard-agnostic -- add "cloudburst" back here to
// resume fetching it in the background.
const HAZARDS = ["thunderstorm"];

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
          error: err.message || "Live fetch failed.",
          // Keep the previously successful result visible if Open-Meteo is
          // rate-limiting the current refresh, instead of blanking the UI.
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

  // Fetch both hazards once on app load, then keep refreshing on a timer.
  // Lives here (not in a page component) so it starts once per app
  // session and isn't torn down/restarted by route navigation.
  //
  // initialFetchStarted guards ONLY the one-time kickoff fetch, not the
  // interval setup below. Reason: React 18 StrictMode (development only)
  // deliberately runs this effect twice -- mount, cleanup, mount again --
  // to surface effects that aren't safe to repeat. setInterval/clearInterval
  // pairing is already safe to run twice (the first interval is cleared
  // before it can ever fire), but fetchHazard() has a real side effect
  // (an actual network call), so without this guard StrictMode caused two
  // concurrent live fetches to fire on every page load.
  const initialFetchStarted = useRef(false);

  useEffect(() => {
    if (!initialFetchStarted.current) {
      initialFetchStarted.current = true;
      HAZARDS.forEach((hazard) => {
        fetchHazard(hazard);
        fetchLog(hazard)
          .then((d) => patchHazard(hazard, { logRows: d.rows }))
          .catch(() => {});
      });
    }

    const interval = setInterval(() => {
      HAZARDS.forEach((hazard) => fetchHazard(hazard));
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
