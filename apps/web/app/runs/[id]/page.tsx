"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { getRun, type RunDetail } from "@/lib/api";
import { benchmarkLabel, benchmarkScope, retrievalIsComplete } from "@/lib/benchmark-scope";
import { CaseEvidence } from "@/components/CaseEvidence";
import { EvaluationStatus } from "@/components/EvaluationStatus";
import { MetricCard } from "@/components/MetricCard";
import { StrategyBadge } from "@/components/StrategyBadge";
import { RunAnnotations } from "@/components/RunAnnotations";

export default function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  return <RunView key={id} id={id} />;
}

function RunView({ id }: { id: string }) {
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getRun(id, controller.signal)
      .then((result) => { if (!controller.signal.aborted) setRun(result); })
      .catch((e: unknown) => { if (!controller.signal.aborted) setError(String(e)); });
    return () => controller.abort();
  }, [id]);

  useEffect(() => {
    if (!run) return;
    const openCase = () => {
      if (!window.location.hash.startsWith("#case-")) return;
      try {
        const element = document.getElementById(decodeURIComponent(window.location.hash.slice(1)));
        if (element instanceof HTMLDetailsElement) { element.open = true; element.scrollIntoView(); }
      } catch { /* An invalid fragment does not prevent inspecting the run. */ }
    };
    openCase();
    window.addEventListener("hashchange", openCase);
    return () => window.removeEventListener("hashchange", openCase);
  }, [run]);

  if (error) return <div className="workflow-error" role="alert">Could not load this run. {error}</div>;
  if (!run) return <div className="empty muted">Loading…</div>;

  const m = run.retrieval_metrics;
  const a = run.answer_metrics;
  const scope = benchmarkScope(run.config);
  const evidence = run.evidence_metrics;
  const eligible = evidence?.eligible_cases !== 0;
  const retrievalValue = (value: number) => eligible ? value.toFixed(3) : "N/A";

  return (
    <>
      <h1 className="page-title">
        Run <span className="num">{run.id.slice(0, 8)}</span>{" "}
        <StrategyBadge strategy={String(run.config.strategy ?? "—")} />
      </h1>
      <p className="page-sub">{new Date(run.created_at).toLocaleString()}</p>
      <div className="workflow-actions run-actions">
        {scope && retrievalIsComplete(run) ? <Link className="row-link" href={`/compare?baseline=${encodeURIComponent(run.id)}`}>Compare this run</Link> : null}
        <a className="row-link" href="#notes">Review notes</a>
        {!retrievalIsComplete(run) ? <Link className="row-link" href="/experiments">Follow evaluation progress</Link> : null}
      </div>

      <div className="benchmark-detail">
        <div className="benchmark-detail__label">Benchmark</div>
        <div>{scope ? benchmarkLabel(scope) : "Legacy or incomplete benchmark scope"}</div>
        {scope ? (
          <details className="benchmark-identity">
            <summary>Benchmark identity</summary>
            <dl>
              <dt>Dataset content</dt><dd>{scope.datasetSha256}</dd>
              <dt>Case set</dt><dd>{scope.caseSetSha256}</dd>
              <dt>Evidence metric</dt><dd>{scope.evidenceMetricVersion}</dd>
              <dt>Ranking metric</dt><dd>{scope.rankingMetricVersion}</dd>
              {typeof run.config.index_id === "string" ? (
                <><dt>Index snapshot</dt><dd>{run.config.index_id}</dd></>
              ) : null}
            </dl>
          </details>
        ) : <p className="benchmark-selector__help">Comparability with other runs is unknown.</p>}
      </div>

      <EvaluationStatus
        evaluation={run.evaluation}
        costUsd={run.cost_usd}
        costComplete={run.cost_complete}
      />

      <div className="section-title">{retrievalIsComplete(run) ? "Retrieval" : "Retrieval · saved cases only"}</div>
      <div className="metric-grid">
        <MetricCard label="Recall@k" value={retrievalValue(m.recall)} accent />
        <MetricCard label="nDCG@k" value={retrievalValue(m.ndcg)} accent />
        <MetricCard label="MRR" value={retrievalValue(m.mrr)} />
        <MetricCard label="Precision@k" value={retrievalValue(m.precision)} />
        <MetricCard label="Evidence recall" value={evidence ? evidence.recall?.toFixed(3) ?? "N/A" : "—"}
          sub={evidence ? `${evidence.eligible_cases} eligible · ${evidence.unanswerable_cases} unanswerable` : "Not recorded"} accent />
        <MetricCard
          label="Retrieval p95"
          value={`${run.latency_p95_ms}ms`}
          sub={`p50 ${run.latency_p50_ms}ms`}
        />
      </div>

      {evidence?.unanswerable_cases ? (
        <p className="benchmark-notice">
          Unanswerable cases are excluded from retrieval-quality averages.
        </p>
      ) : null}

      {a ? (
        <>
          <div className="section-title">
            {run.evaluation === null
              ? "Answer quality (legacy coverage unknown)"
              : "Answer quality (scored cases only)"}
          </div>
          <div className="metric-grid">
            <MetricCard label="Faithfulness" value={a.faithfulness.toFixed(3)} accent />
            <MetricCard label="Relevance" value={a.relevance.toFixed(3)} accent />
            <MetricCard label="Citation correctness" value={a.citation_correctness.toFixed(3)} />
          </div>
        </>
      ) : run.evaluation?.with_answers ? (
        <>
          <div className="section-title">Answer quality</div>
          <div className="answer-metrics-empty">
            {run.evaluation.scored === 0
              ? "No answer-quality aggregate: no cases were scored."
              : "Answer-quality aggregate unavailable for this run."}
          </div>
        </>
      ) : null}

      <div className="section-title">Per-case results</div>
      <div className="case-list">
        {run.cases.map((result) => (
          <CaseEvidence result={result} key={result.case_id} />
        ))}
      </div>
      <RunAnnotations runId={run.id} cases={run.cases} />
    </>
  );
}
