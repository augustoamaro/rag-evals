"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { listRuns, type RunSummary } from "@/lib/api";
import { benchmarkLabel, groupBenchmarkRuns } from "@/lib/benchmark-scope";
import { EvaluationStatus } from "@/components/EvaluationStatus";
import { StrategyBadge } from "@/components/StrategyBadge";
import { TrendChart } from "@/components/TrendChart";

function RunsTable({ runs, label }: { runs: RunSummary[]; label: string }) {
  return (
    <div className="panel">
      <table aria-label={label}>
        <thead>
          <tr>
            <th>Strategy</th>
            <th>k</th>
            <th>Recall</th>
            <th>nDCG</th>
            <th>MRR</th>
            <th>Evidence recall</th>
            <th>Evaluation</th>
            <th>Retrieval p95</th>
            <th>Run</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => {
            const evidence = run.evidence_metrics;
            const eligible = evidence?.eligible_cases !== 0;
            return (
              <tr key={run.id}>
                <td><StrategyBadge strategy={String(run.config.strategy ?? "—")} /></td>
                <td className="num">{String(run.config.k ?? "—")}</td>
                <td className="num">{eligible ? run.retrieval_metrics.recall.toFixed(3) : "N/A"}</td>
                <td className="num">{eligible ? run.retrieval_metrics.ndcg.toFixed(3) : "N/A"}</td>
                <td className="num">{eligible ? run.retrieval_metrics.mrr.toFixed(3) : "N/A"}</td>
                <td className="benchmark-evidence-cell">
                  <span className="num">{evidence ? evidence.recall?.toFixed(3) ?? "N/A" : "—"}</span>
                  <span className="benchmark-evidence-cell__count">
                    {evidence ? `${evidence.eligible_cases} eligible · ${evidence.unanswerable_cases} unanswerable` : "Not recorded"}
                  </span>
                </td>
                <td>
                  <EvaluationStatus evaluation={run.evaluation} costUsd={run.cost_usd}
                    costComplete={run.cost_complete} compact />
                </td>
                <td className="num">{run.latency_p95_ms}ms</td>
                <td>
                  <Link href={`/runs/${run.id}`} className="row-link" aria-label={`View run ${run.id}`}>
                    view →
                  </Link>
                  <time className="benchmark-run-date" dateTime={run.created_at}>
                    {new Date(run.created_at).toLocaleDateString()}
                  </time>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function Home() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const history = useMemo(() => runs ? groupBenchmarkRuns(runs) : null, [runs]);
  const selected = history?.groups.find((group) => group.key === selectedKey)
    ?? history?.groups.find((group) => group.key === history.defaultKey);
  const trendRuns = selected?.runs.filter((run) => run.evidence_metrics?.eligible_cases !== 0) ?? [];

  useEffect(() => {
    const controller = new AbortController();
    listRuns(controller.signal)
      .then((result) => { if (!controller.signal.aborted) setRuns(result); })
      .catch((e: unknown) => { if (!controller.signal.aborted) setError(String(e)); });
    return () => controller.abort();
  }, []);

  return (
    <>
      <h1 className="page-title">Evaluation runs</h1>
      <p className="page-sub">Retrieval and answer quality for each benchmark and language.</p>
      <div className="workflow-actions home-actions">
        <Link href="/experiments" className="btn">Launch an evaluation</Link>
        <Link href="/compare" className="btn btn--secondary">Compare runs</Link>
      </div>

      {error ? (
        <div className="workflow-error" role="alert">Could not load runs. {error}</div>
      ) : runs === null ? (
        <div className="empty muted">Loading…</div>
      ) : runs.length === 0 ? (
        <div className="empty muted">No runs yet. Launch an evaluation to measure retrieval for your benchmark.</div>
      ) : (
        <>
          {selected && history ? (
            <>
              <section className="benchmark-selector" aria-label="Benchmark selection">
                <label htmlFor="benchmark">Benchmark</label>
                <select id="benchmark" className="select" value={selected.key}
                  onChange={(event) => setSelectedKey(event.target.value)}>
                  {history.groups.map((group) => (
                    <option key={group.key} value={group.key}>
                      {benchmarkLabel(group)} · {group.runs.length} {group.runs.length === 1 ? "run" : "runs"}
                    </option>
                  ))}
                </select>
                <p className="benchmark-selector__scope">{benchmarkLabel(selected)}</p>
                <p className="benchmark-selector__help">
                  History includes the same dataset, cases, language, k and metric definitions.
                </p>
              </section>
              <div className="section-title">Recall &amp; nDCG over time</div>
              {trendRuns.length > 1 ? <TrendChart runs={trendRuns} /> : (
                <div className="benchmark-notice">
                  {trendRuns.length === 0
                    ? "Retrieval quality is not applicable: this benchmark has no eligible cases."
                    : "A trend appears after two eligible runs of this benchmark."}
                </div>
              )}
              <div className="section-title">Benchmark runs · {selected.runs.length}</div>
              <RunsTable runs={selected.runs} label="Selected benchmark runs" />
            </>
          ) : null}

          {history?.unscoped.length ? (
            <>
              <div className="section-title">Other recorded runs · {history.unscoped.length}</div>
              <p className="benchmark-notice">
                These runs have incomplete retrieval, or legacy or incomplete benchmark metadata. Open them individually;
                a comparable history is unavailable.
              </p>
              <RunsTable runs={history.unscoped} label="Runs without a comparable benchmark scope" />
            </>
          ) : null}
        </>
      )}
    </>
  );
}
