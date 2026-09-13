/**
 * Typed client for the judgekit API.
 *
 * The types here mirror the response schemas rather than the core Python
 * models, which is the same separation the backend makes: an HTTP response is a
 * contract with another program, and collapsing it into the internal structure
 * turns every refactor into a breaking change.
 */

const API_BASE = process.env.JUDGEKIT_API_URL ?? "http://localhost:8000";

export type RubricRef = {
  id: string;
  version: string;
  fingerprint: string;
};

export type RunSummary = {
  run_id: string;
  status: "queued" | "running" | "completed" | "failed";
  dataset: string;
  dataset_version: string;
  rubric: RubricRef;
  provider: string;
  model: string;
  total: number;
  passed: number;
  failed: number;
  errored: number;
  pass_rate: number;
  mean_score: number;
  cost_usd: number;
  created_at: string;
  finished_at: string | null;
  error: string | null;
};

export type Judgement = {
  case_id: string;
  score: number;
  verdict: "pass" | "fail" | "error";
  reasoning: string;
  flags: string[];
  criterion_scores: Record<string, number>;
  elapsed_ms: number;
  error: string | null;
};

export type RunDetail = RunSummary & { judgements: Judgement[] };

export type RunList = { runs: RunSummary[]; count: number };

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function get<T>(path: string): Promise<T> {
  // no-store, because eval runs change while somebody is watching the page.
  // A cached "queued" that never updates is worse than a slightly slower load.
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new ApiError(detail || response.statusText, response.status);
  }
  return (await response.json()) as T;
}

export function listRuns(params: {
  fingerprint?: string;
  dataset?: string;
  limit?: number;
} = {}): Promise<RunList> {
  const query = new URLSearchParams();
  if (params.fingerprint) query.set("rubric_fingerprint", params.fingerprint);
  if (params.dataset) query.set("dataset", params.dataset);
  query.set("limit", String(params.limit ?? 50));
  return get<RunList>(`/runs?${query}`);
}

export function getRun(runId: string): Promise<RunDetail> {
  return get<RunDetail>(`/runs/${encodeURIComponent(runId)}`);
}

/**
 * Runs that may legitimately be charted alongside this one.
 *
 * Deliberately a server-side concept rather than a client-side filter: the API
 * scopes it to the same dataset and the same rubric fingerprint, so a trend
 * line cannot silently span a rubric change.
 */
export function getHistory(runId: string): Promise<RunList> {
  return get<RunList>(`/runs/${encodeURIComponent(runId)}/history`);
}

export function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function formatWhen(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}
