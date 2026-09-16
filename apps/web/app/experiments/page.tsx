"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  ApiError, cancelJob, getEvaluationOptions, getJob, launchJob, listJobs,
  type EvaluationJob, type EvaluationOptions, type JobRequest, type RunRequest,
} from "@/lib/api";
import { languageLabel } from "@/lib/benchmark-scope";
import { StrategyBadge } from "@/components/StrategyBadge";

const PENDING_KEY = "rag-evals.pending-launch.v1";
const active = (job: EvaluationJob) => ["queued", "running", "cancelling"].includes(job.status);
const defaults: RunRequest = { strategy: "hybrid", k: 10, with_answers: false, split: "smoke", language: "en" };

function rememberPending(request: JobRequest | null) {
  try {
    if (request) sessionStorage.setItem(PENDING_KEY, JSON.stringify(request));
    else sessionStorage.removeItem(PENDING_KEY);
  } catch { /* The in-memory token still protects retries if storage is unavailable. */ }
}

function readPending(): JobRequest | null {
  try {
    const request: unknown = JSON.parse(sessionStorage.getItem(PENDING_KEY) ?? "null");
    if (!request || typeof request !== "object") return null;
    const value = request as Partial<JobRequest>;
    if (typeof value.request_id === "string" && /^[0-9a-f-]{36}$/i.test(value.request_id)
      && typeof value.strategy === "string" && typeof value.k === "number" && Number.isInteger(value.k)
      && value.k >= 1 && value.k <= 50 && typeof value.with_answers === "boolean"
      && typeof value.split === "string" && typeof value.language === "string") return value as JobRequest;
  } catch { /* Invalid or blocked session storage does not prevent a new launch. */ }
  return null;
}

function mergeJob(jobs: EvaluationJob[], next: EvaluationJob): EvaluationJob[] {
  const previous = jobs.find((job) => job.id === next.id);
  if (previous && (previous.updated_at > next.updated_at
    || (!active(previous) && active(next))
    || (previous.status === "cancelling" && ["queued", "running"].includes(next.status)))) return jobs;
  return previous ? jobs.map((job) => job.id === next.id ? next : job) : [next, ...jobs];
}

export default function ExperimentsPage() {
  const [options, setOptions] = useState<EvaluationOptions | null>(null);
  const [jobs, setJobs] = useState<EvaluationJob[]>([]);
  const [jobsLoaded, setJobsLoaded] = useState(false);
  const [config, setConfig] = useState<RunRequest>(defaults);
  const [pending, setPending] = useState<JobRequest | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [cancelling, setCancelling] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  const initialized = useRef(false);
  const pendingRequest = useRef<JobRequest | null>(null);
  const mutation = useRef<AbortController | null>(null);
  const activeJobId = jobs.find(active)?.id ?? null;
  const selection = options?.selections.find((item) => item.split === config.split && item.language === config.language);
  const languages = [...new Set(options?.selections.map((item) => item.language) ?? [])];
  const locked = submitting || pending !== null;

  useEffect(() => () => mutation.current?.abort(), []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    void Promise.allSettled([getEvaluationOptions(controller.signal), listJobs(controller.signal)])
      .then(([optionsResult, jobsResult]) => {
        if (controller.signal.aborted) return;
        const saved = pendingRequest.current ?? readPending();
        const recent = jobsResult.status === "fulfilled" ? jobsResult.value : [];
        const recovered = saved && recent.some((job) => job.request_id === saved.request_id);
        if (recovered) rememberPending(null);
        pendingRequest.current = recovered ? null : saved;
        setPending(recovered ? null : saved);
        if (jobsResult.status === "fulfilled") {
          setJobs((current) => recent.reduce(mergeJob, current).sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, 50));
          setJobsLoaded(true);
        }
        const failures = [optionsResult, jobsResult].flatMap((result) => result.status === "rejected" ? [String(result.reason)] : []);
        if (failures.length) setError(failures.join(". "));
        if (optionsResult.status !== "fulfilled") return;
        const available = optionsResult.value;
        setOptions(available);
        if (!initialized.current) {
          const initial = available.selections.find((item) => item.language === "en" && item.split === available.default_split)
            ?? available.selections.find((item) => item.language === "en") ?? available.selections[0];
          setConfig(saved ?? { ...defaults, split: initial?.split ?? available.default_split,
            language: initial?.language ?? "en", strategy: available.strategies.includes("hybrid") ? "hybrid" : available.strategies[0] ?? "hybrid" });
          initialized.current = true;
        }
      })
      .catch((reason: unknown) => { if (!controller.signal.aborted) setError(String(reason)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    if (!activeJobId) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function poll() {
      let again = true;
      try {
        const next = await getJob(activeJobId!, controller.signal);
        if (controller.signal.aborted) return;
        setJobs((current) => mergeJob(current, next));
        setPollError(null);
        again = active(next);
      } catch (reason) {
        if (!controller.signal.aborted) setPollError(`Progress unavailable; retrying. ${String(reason)}`);
      }
      if (!controller.signal.aborted && again) timer = setTimeout(() => void poll(), 1500);
    }
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [activeJobId]);

  const submit = async () => {
    if (mutation.current || loading || (!pending && (!selection || !options?.worker_available || activeJobId))) return;
    const request: JobRequest = pending ?? { ...config, with_answers: config.with_answers && Boolean(selection?.answers_available), request_id: crypto.randomUUID() };
    const controller = new AbortController();
    mutation.current = controller;
    pendingRequest.current = request;
    setPending(request);
    rememberPending(request);
    setSubmitting(true);
    setError(null);
    try {
      const next = await launchJob(request, controller.signal);
      if (controller.signal.aborted) return;
      setJobs((current) => mergeJob(current, next));
      pendingRequest.current = null;
      setPending(null);
      rememberPending(null);
    } catch (reason) {
      if (controller.signal.aborted) return;
      const definite = reason instanceof ApiError && reason.status < 500 && reason.status !== 408;
      if (definite) { pendingRequest.current = null; setPending(null); rememberPending(null); }
      setError(definite ? String(reason) : `Could not confirm the launch. Retry safely with the same request. ${String(reason)}`);
    } finally {
      if (!controller.signal.aborted) setSubmitting(false);
      if (mutation.current === controller) mutation.current = null;
    }
  };

  const cancel = async (id: string) => {
    if (mutation.current) return;
    const controller = new AbortController();
    mutation.current = controller;
    setCancelling(id);
    setError(null);
    try {
      const next = await cancelJob(id, controller.signal);
      if (!controller.signal.aborted) setJobs((current) => mergeJob(current, next));
    } catch (reason) {
      if (!controller.signal.aborted) setError(`Cancellation could not be confirmed. You can retry. ${String(reason)}`);
    } finally {
      if (!controller.signal.aborted) setCancelling(null);
      if (mutation.current === controller) mutation.current = null;
    }
  };

  const chooseLanguage = (language: string) => {
    const next = options?.selections.find((item) => item.language === language && item.split === config.split)
      ?? options?.selections.find((item) => item.language === language);
    setConfig({ ...config, language, split: next?.split ?? "", with_answers: false });
  };

  return (
    <>
      <h1 className="page-title">Experiments</h1>
      <p className="page-sub">Launch a benchmark, follow saved progress, and inspect the result.</p>
      {error ? <div className="workflow-error" role="alert">{error}</div> : null}
      {loading && !options ? <p className="empty" role="status">Loading experiment options…</p> : null}
      {options ? (
        <form className="experiment-form" onSubmit={(event) => { event.preventDefault(); void submit(); }}>
          <div className="experiment-form__heading">
            <div><h2>New evaluation</h2><p>{options.dataset ? `${options.dataset.name} · ${options.dataset.version}` : "Legacy evaluation dataset"}</p></div>
            {selection ? <span className="case-count">{selection.case_count} cases</span> : null}
          </div>
          <fieldset className="workflow-fields" disabled={locked}>
            <legend className="sr-only">Evaluation configuration</legend>
            <label>Language<select className="select" value={config.language} onChange={(event) => chooseLanguage(event.target.value)}>
              {languages.map((language) => <option key={language} value={language}>{languageLabel(language)}</option>)}
            </select></label>
            <label>Split<select className="select" value={config.split} onChange={(event) => setConfig({ ...config, split: event.target.value, with_answers: false })}>
              {options.selections.filter((item) => item.language === config.language).map((item) => <option key={item.split} value={item.split}>{item.split}</option>)}
            </select></label>
            <label>Strategy<select className="select" value={config.strategy} onChange={(event) => setConfig({ ...config, strategy: event.target.value })}>
              {options.strategies.map((strategy) => <option key={strategy}>{strategy}</option>)}
            </select></label>
            <label>Retrieval depth (k)<input className="select" type="number" min="1" max="50" required value={config.k}
              onChange={(event) => setConfig({ ...config, k: Number(event.target.value) })} /></label>
          </fieldset>
          <p className="benchmark-notice">English and Portuguese are evaluated separately. Smoke and dev results are separate from the published test split.</p>
          <div className="answer-option">
            <label><input type="checkbox" checked={config.with_answers && Boolean(selection?.answers_available)}
              disabled={locked || !selection?.answers_available} aria-describedby="answer-availability"
              onChange={(event) => setConfig({ ...config, with_answers: event.target.checked })} />Generate and judge answers</label>
            <p id="answer-availability">{selection?.answers_available
              ? "Optional: calls the configured model and records provider usage and cost."
              : selection?.answer_unavailable_reason ?? "Answer evaluation is unavailable for this selection."}</p>
          </div>
          {!options.worker_available ? <p className="workflow-warning">The evaluation worker is unavailable. Refresh after the service is ready.</p> : null}
          {activeJobId ? <p className="benchmark-notice">An evaluation is active. Its progress is shown below.</p> : null}
          {pending && !submitting ? <p className="workflow-warning">A launch is unconfirmed. Retry this saved configuration to recover its job before starting another evaluation.</p> : null}
          <div className="workflow-actions">
            <button className="btn" type="submit" disabled={submitting || loading || (!pending && (!selection || !options.worker_available || Boolean(activeJobId)))}>
              {submitting ? "Launching…" : pending ? "Retry launch" : "Launch evaluation"}
            </button>
            <button className="btn btn--secondary" type="button" disabled={loading || submitting} onClick={() => setRefresh((value) => value + 1)}>Refresh jobs</button>
          </div>
        </form>
      ) : !loading ? <button className="btn" onClick={() => setRefresh((value) => value + 1)}>Retry loading</button> : null}

      <div className="section-title">Recent jobs</div>
      {pollError ? <p role="status" className="workflow-warning">{pollError}</p> : null}
      {!jobs.length && !loading && jobsLoaded ? <p className="benchmark-notice">No evaluations have been launched yet. Choose a benchmark above to begin.</p> : null}
      <div className="job-list">
        {jobs.map((job) => (
          <article className="job" key={job.id} aria-label={`Evaluation ${job.id}`}>
            <div className="job__heading">
              <div><StrategyBadge strategy={job.config.strategy} /><span className={`job-status job-status--${job.status}`}>{job.status}</span></div>
              <time dateTime={job.created_at}>{new Date(job.created_at).toLocaleString()}</time>
            </div>
            <p className="job__scope">{job.config.split} · {languageLabel(job.config.language)} · k={job.config.k} · {job.config.with_answers ? "Answers and judging" : "Retrieval only"}</p>
            <div className="job__progress" role="status">
              <span className="num">{job.completed} / {job.total} cases</span>
              <span>{job.stage ?? "Waiting"}{job.case_id ? ` · ${job.case_id}` : ""}</span>
            </div>
            <progress max={Math.max(1, job.total)} value={job.completed} aria-label={`Completed cases for ${job.id}`} />
            {job.status === "cancelling" ? <p className="benchmark-notice">Cancellation requested. An in-flight call can finish; its observed result and usage are retained.</p> : null}
            {job.error_type ? <p className="workflow-warning">{job.error_type}</p> : null}
            {job.status === "cancelled" ? <p className="benchmark-notice">Saved progress is available for inspection and excluded from comparisons.</p> : null}
            <div className="workflow-actions">
              {job.run_id ? <Link className="row-link" href={`/runs/${encodeURIComponent(job.run_id)}`}>Open run</Link> : <span className="benchmark-notice">Waiting for the first saved result.</span>}
              {active(job) ? <button className="btn btn--secondary" disabled={cancelling === job.id || job.status === "cancelling"} onClick={() => void cancel(job.id)}>
                {cancelling === job.id || job.status === "cancelling" ? "Cancelling…" : "Cancel evaluation"}
              </button> : null}
            </div>
          </article>
        ))}
      </div>
    </>
  );
}
