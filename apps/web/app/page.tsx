"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { listRuns, type RunSummary } from "@/lib/api";
import { StrategyBadge } from "@/components/StrategyBadge";
import { TrendChart } from "@/components/TrendChart";

export default function Home() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listRuns()
      .then(setRuns)
      .catch((e: unknown) => setError(String(e)));
  }, []);

  return (
    <>
      <h1 className="page-title">Evaluation runs</h1>
      <p className="page-sub">
        Retrieval and answer-quality metrics across runs of the RAG pipeline.
      </p>

      {error ? (
        <div className="empty">
          Could not reach the API. Is it running at the configured URL?
        </div>
      ) : runs === null ? (
        <div className="empty muted">Loading…</div>
      ) : runs.length === 0 ? (
        <div className="empty muted">
          No runs yet. Run <code>rag eval</code> or POST <code>/evals/run</code>.
        </div>
      ) : (
        <>
          {runs.length > 1 ? (
            <>
              <div className="section-title">Recall &amp; nDCG over time</div>
              <TrendChart runs={runs} />
            </>
          ) : null}

          <div className="section-title">All runs</div>
          <div className="panel">
            <table>
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>k</th>
                  <th>Recall</th>
                  <th>nDCG</th>
                  <th>MRR</th>
                  <th>Cost</th>
                  <th>p95</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.id}>
                    <td>
                      <StrategyBadge strategy={String(r.config.strategy ?? "—")} />
                    </td>
                    <td className="num">{String(r.config.k ?? "—")}</td>
                    <td className="num">{r.retrieval_metrics.recall.toFixed(3)}</td>
                    <td className="num">{r.retrieval_metrics.ndcg.toFixed(3)}</td>
                    <td className="num">{r.retrieval_metrics.mrr.toFixed(3)}</td>
                    <td className="num">${r.cost_usd.toFixed(4)}</td>
                    <td className="num">{r.latency_p95_ms}ms</td>
                    <td>
                      <Link href={`/runs/${r.id}`} className="row-link">
                        view →
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </>
  );
}
