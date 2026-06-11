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

export interface RunSummary {
  id: string;
  created_at: string;
  config: Record<string, unknown>;
  retrieval_metrics: RunMetrics;
  answer_metrics: AnswerMetrics | null;
  cost_usd: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
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
  latency_ms: number;
}

export interface RunDetail extends RunSummary {
  cases: CaseResult[];
}

export interface RetrievedChunk {
  id: string;
  text: string;
  score: number;
  rank: number;
}

export interface QueryResponse {
  answer: string;
  citations: { chunk_id: string }[];
  chunks: RetrievedChunk[];
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return (await res.json()) as T;
}

export const listRuns = () => getJson<RunSummary[]>("/evals/runs");
export const getRun = (id: string) => getJson<RunDetail>(`/evals/runs/${id}`);

export async function query(
  question: string,
  k: number,
  strategy: string,
): Promise<QueryResponse> {
  const res = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question, k, strategy }),
  });
  if (!res.ok) throw new Error(`query → ${res.status}`);
  return (await res.json()) as QueryResponse;
}
