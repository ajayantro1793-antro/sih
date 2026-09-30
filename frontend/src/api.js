const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function request(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // response wasn't JSON -- fall back to statusText
    }
    throw new Error(detail);
  }
  return res.json();
}

export function fetchLivePrediction(hazard) {
  return request(`/api/live/${hazard}`);
}

export function fetchLog(hazard, limit = 200) {
  return request(`/api/log/${hazard}?limit=${limit}`);
}

export function fetchMetadata(hazard) {
  return request(`/api/metadata/${hazard}`);
}

export function fetchConfig() {
  return request(`/api/config`);
}
