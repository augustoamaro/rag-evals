"use client";

import { use, useEffect, useState } from "react";
import { getRun, type RunDetail } from "@/lib/api";
import { MetricCard } from "@/components/MetricCard";
import { StrategyBadge } from "@/components/StrategyBadge";

export default function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getRun(id)
      .then(setRun)
      .catch((e: unknown) => setError(String(e)));
  }, [id]);

  if (error) return <div className="empty">Run not found.</div>;
  if (!run) return <div className="empty muted">Loading…</div>;

  const m = run.retrieval_metrics;
  const a = run.answer_metrics;

  return (
    <>
      <h1 className="page-title">
        Run <span className="num">{run.id.slice(0, 8)}</span>{" "}
        <StrategyBadge strategy={String(run.config.strategy ?? "—")} />
      </h1>
      <p className="page-sub">{new Date(run.created_at).toLocaleString()}</p>

      <div className="section-title">Retrieval</div>
      <div className="metric-grid">
        <MetricCard label="Recall@k" value={m.recall.toFixed(3)} accent />
        <MetricCard label="nDCG@k" value={m.ndcg.toFixed(3)} accent />
        <MetricCard label="MRR" value={m.mrr.toFixed(3)} />
        <MetricCard label="Precision@k" value={m.precision.toFixed(3)} />
        <MetricCard label="Latency p95" value={`${run.latency_p95_ms}ms`} sub={`p50 ${run.latency_p50_ms}ms`} />
      </div>

      {a ? (
        <>
          <div className="section-title">Answer quality (LLM-as-judge)</div>
          <div className="metric-grid">
            <MetricCard label="Faithfulness" value={a.faithfulness.toFixed(3)} accent />
            <MetricCard label="Relevance" value={a.relevance.toFixed(3)} accent />
            <MetricCard label="Citation correctness" value={a.citation_correctness.toFixed(3)} />
            <MetricCard label="Cost" value={`$${run.cost_usd.toFixed(4)}`} />
          </div>
        </>
      ) : null}

      <div className="section-title">Per-case results</div>
      <div className="panel">
        <table>
          <thead>
            <tr>
              <th>Case</th>
              <th>Recall</th>
              <th>nDCG</th>
              <th>MRR</th>
              <th>Top retrieved</th>
            </tr>
          </thead>
          <tbody>
            {run.cases.map((c) => (
              <tr key={c.case_id}>
                <td className="num">{c.case_id}</td>
                <td className="num">{c.recall.toFixed(2)}</td>
                <td className="num">{c.ndcg.toFixed(2)}</td>
                <td className="num">{c.mrr.toFixed(2)}</td>
                <td className="num muted">{c.retrieved_ids.slice(0, 3).join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
