/**
 * Server-side API client.
 *
 * All data fetching happens in React Server Components, so the browser never
 * talks to the API directly and no credentials reach the client bundle. Each
 * call declares its own caching behaviour: scores are computed live and must
 * never be cached, whereas the feature schema and connector catalogue are
 * effectively static.
 */

import { DEMO_CONTEXT } from "./config";
import type {
  Coefficient, ComparatorResponse, ConnectorHealth, EvidenceMatrix, EvidenceOut,
  FeatureSpec, HealthOut, PortfolioResponse, ProgramDetail, ScenarioComparison,
  ScenarioLever, ScoreOut, TrackSummary, ValidationReport,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";


export class ApiError extends Error {
  constructor(readonly status: number, readonly path: string, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

type FetchOptions = {
  revalidate?: number | false;
  method?: "GET" | "POST";
  body?: unknown;
};

async function request<T>(path: string, options: FetchOptions = {}): Promise<T> {
  const { revalidate = 0, method = "GET", body } = options;
  const url = `${BASE}${path}`;

  const res = await fetch(url, {
    method,
    headers: { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    // `no-store` for live computations; a positive number for reference data.
    ...(revalidate === false || method === "POST"
      ? { cache: "no-store" as const }
      : { next: { revalidate } }),
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const payload = await res.json();
      detail = payload?.detail ?? detail;
    } catch {
      /* non-JSON error body; keep the status text */
    }
    throw new ApiError(res.status, path, detail);
  }
  return (await res.json()) as T;
}

/** Returns null instead of throwing, for optional panels that may be absent. */
export async function tryRequest<T>(path: string, options?: FetchOptions): Promise<T | null> {
  try {
    return await request<T>(path, options);
  } catch {
    return null;
  }
}

const q = (params: Record<string, string | number | boolean | undefined | null>) => {
  const search = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") search.set(k, String(v));
  }
  const s = search.toString();
  return s ? `?${s}` : "";
};

export const api = {
  health: () => tryRequest<HealthOut>("/api/v1/health", { revalidate: 0 }),

  portfolio: (params: {
    limit?: number; offset?: number; score?: boolean; therapeutic_area?: string;
    modality?: string; target?: string; project_id?: string;
    demo_context?: boolean;
  } = {}) =>
    request<PortfolioResponse>(
      `/api/v1/portfolio${q({ demo_context: DEMO_CONTEXT, ...params })}`,
      { revalidate: false },
    ),

  programs: (params: { q?: string; target?: string; modality?: string; limit?: number } = {}) =>
    request<TrackSummary[]>(`/api/v1/programs${q(params)}`, { revalidate: 0 }),

  program: (id: string) => request<ProgramDetail>(`/api/v1/programs/${id}`, { revalidate: 0 }),

  evidence: (id: string, params: { as_of?: string; include_post_cutoff?: boolean } = {}) =>
    request<EvidenceOut[]>(`/api/v1/programs/${id}/evidence${q(params)}`, { revalidate: 0 }),

  evidenceMatrix: (id: string) =>
    request<EvidenceMatrix>(`/api/v1/programs/${id}/evidence-matrix`, { revalidate: 0 }),

  pkpd: (id: string) => tryRequest<any>(`/api/v1/programs/${id}/pkpd`, { revalidate: 0 }),

  targetAssessment: (id: string) =>
    tryRequest<any>(`/api/v1/programs/${id}/target-assessment`, { revalidate: 0 }),

  safety: (id: string) => tryRequest<any>(`/api/v1/programs/${id}/safety`, { revalidate: 0 }),

  /**
   * Scores one track.
   *
   * `demoContext` must be passed explicitly to obtain a number from a model
   * trained on synthetic data; the backend refuses otherwise, and there is no
   * server-side default-on. This surface is a demonstration deployment and
   * labels itself as one on every page, so it opts in — but the opt-in is
   * visible here rather than hidden in the backend.
   */
  score: (id: string, asOf?: string, demoContext = DEMO_CONTEXT) =>
    request<ScoreOut>(
      `/api/v1/programs/${id}/score${q({ as_of: asOf, demo_context: demoContext })}`,
      { revalidate: false },
    ),

  comparators: (id: string, limit = 8) =>
    request<ComparatorResponse>(`/api/v1/programs/${id}/comparators${q({ limit })}`, {
      revalidate: 0,
    }),

  // Contract-describing endpoints are NOT cached.
  //
  // The lever catalogue, the feature schema, the endpoint definitions and the
  // connector catalogue all describe the shape of the API the UI is currently
  // talking to. Caching them survives a backend deploy, after which the
  // interface renders against a contract the server no longer serves — and it
  // does so silently, because every field the UI reads is still present, just
  // stale. During development this collapsed the adverse-scenario group into
  // the improvement group for an hour: the cached payload predated the
  // `direction` field, every lever read as `undefined`, and the filter put all
  // fourteen into one bucket with no error anywhere.
  //
  // These payloads are a few kilobytes each and the API is a local analytical
  // service, so caching buys nothing worth that failure mode.
  levers: () =>
    request<{ levers: ScenarioLever[]; note: string }>("/api/v1/scenarios/levers", {
      revalidate: false,
    }),

  runScenario: (id: string, leverKeys: string[], overrides: Record<string, number> = {}) =>
    request<ScenarioComparison>(`/api/v1/programs/${id}/scenarios`, {
      method: "POST",
      body: { lever_keys: leverKeys, overrides },
    }),

  validation: (version?: string) =>
    request<ValidationReport>(`/api/v1/model/validation${q({ version })}`, { revalidate: 0 }),

  coefficients: () =>
    request<Coefficient[]>("/api/v1/model/coefficients", { revalidate: 0 }),

  modelVersions: () => request<any[]>("/api/v1/model/versions", { revalidate: 0 }),

  featureSchema: () => request<FeatureSpec[]>("/api/v1/model/features", { revalidate: false }),

  endpoints: () => request<any>("/api/v1/model/endpoints", { revalidate: false }),

  cohorts: () => request<any[]>("/api/v1/model/cohorts", { revalidate: 0 }),

  connectors: () =>
    request<ConnectorHealth[]>("/api/v1/provenance/connectors", { revalidate: 0 }),

  connectorCatalogue: () =>
    request<any>("/api/v1/provenance/connectors/catalogue", { revalidate: false }),

  ingestionRuns: () =>
    request<any[]>("/api/v1/provenance/ingestion-runs", { revalidate: 0 }),

  evidenceProvenance: (id: string) =>
    request<any>(`/api/v1/provenance/evidence/${id}`, { revalidate: 0 }),

  scoreLineage: (id: string) =>
    request<any>(`/api/v1/provenance/scores/${id}/lineage`, { revalidate: 0 }),
};
