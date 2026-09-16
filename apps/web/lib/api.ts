export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface RunMetrics {
  recall: number;
  precision: number;
  mrr: number;
  ndcg: number;
}

export interface AnswerMetrics {
  faithfulness: number;
  relevance: number;
  citation_correctness: number;
}

export interface EvaluationSummary {
  status: "running" | "completed" | "partial" | "failed" | "cancelled";
  with_answers: boolean;
  attempted: number;
  scored: number;
  failed: number;
  retrieval_complete?: boolean;
  error_type?: string | null;
}

export interface EvidenceMetrics {
  recall: number | null;
  eligible_cases: number;
  unanswerable_cases: number;
}

export interface SliceMetrics {
  count: number;
  eligible_cases: number;
  recall: number | null;
  precision: number | null;
  mrr: number | null;
  ndcg: number | null;
  evidence_recall: number | null;
}

export interface RunSummary {
  id: string;
  created_at: string;
  config: Record<string, unknown>;
  retrieval_metrics: RunMetrics;
  answer_metrics: AnswerMetrics | null;
  cost_usd: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  git_sha: string | null;
  evaluation: EvaluationSummary | null;
  cost_complete: boolean | null;
  evidence_metrics?: EvidenceMetrics | null;
  slices?: Record<string, SliceMetrics> | null;
}

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
}

export interface StageResult {
  status: "skipped" | "succeeded" | "failed";
  usage: TokenUsage | null;
  cost_usd: number | null;
  latency_ms: number;
  error_type: string | null;
}

export interface StoredChunk {
  id: string;
  document_id: string;
  ordinal: number;
  text: string;
  token_count: number;
  word_start?: number | null;
  word_end?: number | null;
}

export interface ScoreProvenance {
  score: number | null;
  score_type?: string | null;
  score_components?: Record<string, number>;
}

export interface RetrievedContext extends ScoreProvenance {
  chunk: StoredChunk;
  rank: number;
}

export interface CaseResult {
  case_id: string;
  strategy: string;
  retrieved_ids: string[];
  recall: number;
  precision: number;
  mrr: number;
  ndcg: number;
  answer: string | null;
  judge_scores: Record<string, unknown> | null;
  cost_usd: number;
  latency_ms: number; // retrieval only
  answer_ms: number; // generation + judging (0 when the answer track is off)
  generation: StageResult | null;
  judging: StageResult | null;
  cost_complete: boolean | null;
  question: string | null;
  reference_answer: string | null;
  retrieved_context: RetrievedContext[] | null;
  evidence_recall?: number | null;
  covered_evidence_ids?: string[] | null;
  evidence_ids?: string[] | null;
  split?: string | null;
  tags?: string[] | null;
  language?: string | null;
  answerable?: boolean | null;
  group_id?: string | null;
}

export interface RunDetail extends RunSummary {
  cases: CaseResult[];
}

export interface RetrievedChunk extends ScoreProvenance {
  id: string;
  text: string;
  rank: number;
}

export interface QueryResponse {
  answer: string;
  citations: { chunk_id: string }[];
  chunks: RetrievedChunk[];
}

export interface EvaluationOptions {
  dataset: { name: string; version: string; index_id: string | null } | null;
  default_split: string;
  default_language: string;
  strategies: string[];
  selections: {
    split: string;
    language: string;
    case_count: number;
    answers_available: boolean;
    answer_unavailable_reason: string | null;
  }[];
  worker_available: boolean;
}

export interface RunRequest {
  strategy: string;
  k: number;
  with_answers: boolean;
  split: string;
  language: string;
}

export interface JobRequest extends RunRequest {
  request_id: string;
}

export type JobStatus = "queued" | "running" | "cancelling" | "completed" | "partial" | "failed" | "cancelled";

export interface EvaluationJob {
  id: string;
  run_id: string | null;
  request_id: string;
  config: RunRequest;
  status: JobStatus;
  total: number;
  completed: number;
  stage: string | null;
  case_id: string | null;
  error_type: string | null;
  created_at: string;
  updated_at: string;
}

export interface PairedMetric {
  n: number;
  baseline: number;
  candidate: number;
  mean_delta: number;
  ci_low: number;
  ci_high: number;
}

export interface RunComparison {
  baseline_id: string;
  candidate_id: string;
  compatible: boolean;
  passed: boolean;
  failures: string[];
  metrics: Record<string, PairedMetric>;
  slices: Record<string, Record<string, PairedMetric>>;
  cases: {
    case_id: string;
    question: string | null;
    answerable: boolean;
    tags: string[];
    metrics: Record<string, { baseline: number | null; candidate: number | null; delta: number | null }>;
  }[];
}

export interface RunAnnotation {
  case_id: string | null;
  note: string;
  revision: number;
  updated_at: string;
}

export class ApiError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body: unknown = await res.json();
      if (body && typeof body === "object" && "detail" in body) {
        if (typeof body.detail === "string") detail = body.detail;
        else if (Array.isArray(body.detail)) {
          detail = body.detail.map((item: { msg?: string }) => item.msg ?? "Invalid request").join("; ");
        }
      }
    } catch { /* Non-JSON error responses still retain the HTTP status. */ }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

function writeJson<T>(path: string, body: unknown, signal?: AbortSignal, method = "POST") {
  return requestJson<T>(path, {
    method, signal, headers: { "content-type": "application/json" }, body: JSON.stringify(body),
  });
}

export const listRuns = (signal?: AbortSignal) => requestJson<RunSummary[]>("/evals/runs", { signal });
export const getRun = (id: string, signal?: AbortSignal) => requestJson<RunDetail>(`/evals/runs/${encodeURIComponent(id)}`, { signal });
export const getEvaluationOptions = (signal?: AbortSignal) => requestJson<EvaluationOptions>("/evals/options", { signal });
export const listJobs = (signal?: AbortSignal) => requestJson<EvaluationJob[]>("/evals/jobs", { signal });
export const getJob = (id: string, signal?: AbortSignal) => requestJson<EvaluationJob>(`/evals/jobs/${encodeURIComponent(id)}`, { signal });
export const launchJob = (request: JobRequest, signal?: AbortSignal) => writeJson<EvaluationJob>("/evals/jobs", request, signal);
export const cancelJob = (id: string, signal?: AbortSignal) => writeJson<EvaluationJob>(`/evals/jobs/${encodeURIComponent(id)}/cancel`, {}, signal);
export const compareRuns = (baseline: string, candidate: string, signal?: AbortSignal) =>
  requestJson<RunComparison>(`/evals/compare?${new URLSearchParams({ baseline, candidate })}`, { signal });
export const getAnnotations = (id: string, signal?: AbortSignal) => requestJson<RunAnnotation[]>(`/evals/runs/${encodeURIComponent(id)}/annotations`, { signal });
export const saveAnnotation = (id: string, annotation: Pick<RunAnnotation, "case_id" | "note" | "revision">, signal?: AbortSignal) =>
  writeJson<RunAnnotation>(`/evals/runs/${encodeURIComponent(id)}/annotations`, annotation, signal, "PUT");

export async function query(
  question: string,
  k: number,
  strategy: string,
  signal?: AbortSignal,
): Promise<QueryResponse> {
  return writeJson<QueryResponse>("/query", { question, k, strategy }, signal);
}
