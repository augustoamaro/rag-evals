"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { compareRuns, listRuns, type PairedMetric, type RunComparison, type RunSummary } from "@/lib/api";
import { benchmarkLabel, benchmarkScope, groupBenchmarkRuns } from "@/lib/benchmark-scope";

const metricNames: Record<string, string> = {
  recall: "Recall@k", precision: "Precision@k", mrr: "MRR", ndcg: "nDCG@k", evidence_recall: "Evidence recall",
};
const metricLabel = (name: string) => Object.hasOwn(metricNames, name) ? metricNames[name] : name;
const signed = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
const runLabel = (run: RunSummary) => `${run.id.slice(0, 12)} · ${String(run.config.strategy ?? "unknown")} · ${String(run.config.split)} / ${String(run.config.language)} · ${new Date(run.created_at).toLocaleString()}`;

function MetricsTable({ metrics, label }: { metrics: Record<string, PairedMetric>; label: string }) {
  return <div className="panel"><table className="comparison-table" aria-label={label}>
    <thead><tr><th scope="col">Metric</th><th scope="col">Cases</th><th scope="col">Baseline</th><th scope="col">Candidate</th><th scope="col">Paired delta</th><th scope="col">95% interval</th></tr></thead>
    <tbody>{Object.entries(metrics).map(([name, metric]) => <tr key={name}>
      <th scope="row">{metricLabel(name)}</th><td className="num">{metric.n}</td>
      <td className="num">{metric.baseline.toFixed(3)}</td><td className="num">{metric.candidate.toFixed(3)}</td>
      <td className={`num ${metric.mean_delta < 0 ? "delta-negative" : metric.mean_delta > 0 ? "delta-positive" : ""}`}>{signed(metric.mean_delta)}</td>
      <td className="num">{signed(metric.ci_low)} to {signed(metric.ci_high)}</td>
    </tr>)}</tbody>
  </table></div>;
}

export default function ComparePage() {
  return <Suspense fallback={<p className="empty" role="status">Loading comparison…</p>}><CompareView /></Suspense>;
}

function CompareView() {
  const searchParams = useSearchParams();
  const baselineUrl = searchParams.get("baseline") ?? "";
  const candidateUrl = searchParams.get("candidate") ?? "";
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [selectionReady, setSelectionReady] = useState(false);
  const [baseline, setBaseline] = useState(baselineUrl);
  const [candidate, setCandidate] = useState(candidateUrl);
  const [requested, setRequested] = useState<{ baseline: string; candidate: string; attempt: number } | null>(
    baselineUrl && candidateUrl ? { baseline: baselineUrl, candidate: candidateUrl, attempt: 0 } : null,
  );
  const [report, setReport] = useState<RunComparison | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  const history = useMemo(() => groupBenchmarkRuns(runs ?? []), [runs]);
  const comparable = history.groups.flatMap((group) => group.runs);
  const group = history.groups.find((item) => item.runs.some((run) => run.id === baseline));
  const candidates = group?.runs.filter((run) => run.id !== baseline) ?? [];
  const canCompare = candidates.some((run) => run.id === candidate);

  useEffect(() => {
    const controller = new AbortController();
    setLoadError(null);
    listRuns(controller.signal).then((result) => { if (!controller.signal.aborted) setRuns(result); })
      .catch((reason: unknown) => { if (!controller.signal.aborted) setLoadError(String(reason)); });
    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    if (!runs) return;
    const grouped = groupBenchmarkRuns(runs);
    const initialGroup = grouped.groups.find((item) => item.runs.some((run) => run.id === baselineUrl))
      ?? grouped.groups.find((item) => item.key === grouped.defaultKey);
    const initialBaseline = baselineUrl || initialGroup?.runs[1]?.id || initialGroup?.runs[0]?.id || "";
    const initialCandidate = candidateUrl || initialGroup?.runs.find((run) => run.id !== initialBaseline)?.id || "";
    setBaseline(initialBaseline);
    setCandidate(initialCandidate);
    setSelectionReady(true);
    if (baselineUrl && candidateUrl) setRequested((previous) =>
      previous?.baseline === baselineUrl && previous.candidate === candidateUrl
        ? previous : { baseline: baselineUrl, candidate: candidateUrl, attempt: 0 });
    else setRequested(null);
  }, [runs, baselineUrl, candidateUrl]);

  useEffect(() => {
    setReport(null);
    setError(null);
    if (!requested) { setLoading(false); return; }
    const controller = new AbortController();
    setLoading(true);
    compareRuns(requested.baseline, requested.candidate, controller.signal)
      .then((result) => { if (!controller.signal.aborted) setReport(result); })
      .catch((reason: unknown) => { if (!controller.signal.aborted) setError(String(reason)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [requested]);

  const chooseBaseline = (id: string) => {
    const nextGroup = history.groups.find((item) => item.runs.some((run) => run.id === id));
    setBaseline(id);
    setCandidate(nextGroup?.runs.find((run) => run.id !== id)?.id ?? "");
    setRequested(null);
  };

  const submit = () => {
    if (!canCompare) return;
    window.history.replaceState(null, "", `/compare?${new URLSearchParams({ baseline, candidate })}`);
    setRequested((previous) => ({ baseline, candidate, attempt: (previous?.attempt ?? 0) + 1 }));
  };

  const scope = group ? benchmarkScope(group.runs[0]!.config) : null;
  const metricKeys = report ? Object.keys(report.metrics) : [];

  return (
    <>
      <h1 className="page-title">Compare runs</h1>
      <p className="page-sub">Paired retrieval changes for identical cases, language, and metric definitions.</p>
      {loadError ? <div className="workflow-error" role="alert">Could not load runs. {loadError}<button className="btn btn--secondary" onClick={() => setRefresh((value) => value + 1)}>Retry loading</button></div> : null}
      {!selectionReady && !loadError ? <p className="empty" role="status">Loading runs…</p> : null}
      {runs && selectionReady ? (
        <form className="compare-form" onSubmit={(event) => { event.preventDefault(); submit(); }}>
          <div className="workflow-fields workflow-fields--pair">
            <label>Baseline run<select className="select" value={baseline} onChange={(event) => chooseBaseline(event.target.value)}>
              {!comparable.some((run) => run.id === baseline) ? <option value={baseline} disabled>{baseline ? `${baseline} · unavailable for selection` : "No comparable runs"}</option> : null}
              {comparable.map((run) => <option key={run.id} value={run.id}>{runLabel(run)}</option>)}
            </select></label>
            <label>Candidate run<select className="select" value={candidate} onChange={(event) => { setCandidate(event.target.value); setRequested(null); }}>
              {!candidates.some((run) => run.id === candidate) ? <option value={candidate} disabled>{candidate ? `${candidate} · incompatible selection` : "No matching candidate"}</option> : null}
              {candidates.map((run) => <option key={run.id} value={run.id}>{runLabel(run)}</option>)}
            </select></label>
          </div>
          {scope ? <p className="benchmark-selector__scope">{benchmarkLabel(scope)}</p> : null}
          <p className="benchmark-notice">Candidates have the same benchmark scope and complete retrieval. Running, cancelled, and incomplete prefixes are excluded.</p>
          <button className="btn" type="submit" disabled={!canCompare}>Compare runs</button>
          {!canCompare && !candidateUrl ? <p className="benchmark-notice">A second complete run of the same benchmark is needed. <Link className="row-link" href="/experiments">Launch an evaluation</Link></p> : null}
        </form>
      ) : null}
      {loading ? <p className="benchmark-notice" role="status">Comparing paired cases…</p> : null}
      {error ? <div className="workflow-error" role="alert">Could not compare these runs. {error}</div> : null}
      {report ? (
        <>
          <div className="workflow-actions comparison-links">
            <Link className="row-link" href={`/runs/${encodeURIComponent(report.baseline_id)}`} aria-label="Open baseline run">Baseline {report.baseline_id.slice(0, 12)}</Link>
            <Link className="row-link" href={`/runs/${encodeURIComponent(report.candidate_id)}`} aria-label="Open candidate run">Candidate {report.candidate_id.slice(0, 12)}</Link>
          </div>
          {!report.compatible ? (
            <div className="workflow-error" role="alert"><strong>These runs cannot be compared.</strong><ul>{report.failures.map((reason, index) => <li key={index}>{reason}</li>)}</ul></div>
          ) : (
            <>
              <div className={report.passed ? "comparison-check comparison-check--passed" : "comparison-check"}>
                <strong>{report.passed ? "Regression checks passed" : "Regression checks failed"}</strong>
                {report.failures.length ? <ul>{report.failures.map((reason, index) => <li key={index}>{reason}</li>)}</ul> : null}
              </div>
              <h2 className="section-title">Aggregate retrieval comparison</h2>
              <p className="benchmark-notice">Delta = candidate − baseline. Positive values improve retrieval. Intervals are paired 95% bootstrap intervals; this comparison does not measure answer quality.</p>
              <MetricsTable metrics={report.metrics} label="Aggregate retrieval comparison" />
              <h2 className="section-title">Slices</h2>
              <div className="comparison-slices">{Object.entries(report.slices).map(([name, metrics]) => (
                <details key={name} className="comparison-slice"><summary>{name}</summary><MetricsTable metrics={metrics} label={`${name} retrieval comparison`} /></details>
              ))}</div>
              <h2 className="section-title">Per-case changes</h2>
              <p className="benchmark-notice">Open either run’s stored evidence to inspect a change.</p>
              <div className="panel"><table className="case-comparison-table" aria-label="Per-case retrieval changes">
                <thead><tr><th scope="col">Case and evidence</th>{metricKeys.map((name) => <th scope="col" key={name}>{metricLabel(name)} delta</th>)}</tr></thead>
                <tbody>{report.cases.map((result) => <tr key={result.case_id}>
                  <td><div className="case-comparison__id">{result.case_id}</div><div className="case-comparison__question">{result.question ?? "Question not recorded"}</div>
                    <div className="case-comparison__tags">{result.tags.join(", ")}</div>
                    <div className="workflow-actions"><Link className="row-link" aria-label={`Baseline evidence for ${result.case_id}`} href={`/runs/${encodeURIComponent(report.baseline_id)}#case-${encodeURIComponent(result.case_id)}`}>Baseline evidence</Link>
                      <Link className="row-link" aria-label={`Candidate evidence for ${result.case_id}`} href={`/runs/${encodeURIComponent(report.candidate_id)}#case-${encodeURIComponent(result.case_id)}`}>Candidate evidence</Link></div>
                  </td>
                  {!result.answerable ? <td colSpan={Math.max(1, metricKeys.length)} className="benchmark-notice">Unanswerable · retrieval quality not applicable</td> : metricKeys.map((name) => {
                    const metric = result.metrics[name];
                    return <td key={name} className={`num ${metric?.delta != null && metric.delta < 0 ? "delta-negative" : metric?.delta != null && metric.delta > 0 ? "delta-positive" : ""}`}>
                      {metric?.delta == null ? "N/A" : signed(metric.delta)}
                      {metric ? <span className="case-comparison__values">{metric.baseline?.toFixed(3) ?? "N/A"} → {metric.candidate?.toFixed(3) ?? "N/A"}</span> : null}
                    </td>;
                  })}
                </tr>)}</tbody>
              </table></div>
            </>
          )}
        </>
      ) : null}
    </>
  );
}
