import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RunComparison, RunSummary } from "@/lib/api";
import ComparePage from "./page";

vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams(window.location.search) }));
const run = (id: string, language = "en", created = "2026-09-16T12:00:00Z"): RunSummary => ({
  id, created_at: created, config: {
    schema_version: 2, dataset_name: "Engineering", dataset_version: "2026.1", dataset_sha256: "a".repeat(64),
    case_set_sha256: "b".repeat(64), split: "smoke", language, k: 10, retrieval_level: "chunk",
    evidence_metric_version: "evidence-v1", ranking_metric_version: "ranking-v1", strategy: "hybrid",
  },
  retrieval_metrics: { recall: 0.8, precision: 0.5, ndcg: 0.7, mrr: 0.9 }, answer_metrics: null,
  cost_usd: 0, cost_complete: true, latency_p50_ms: 10, latency_p95_ms: 20, git_sha: null,
  evaluation: { status: "completed", retrieval_complete: true, with_answers: false, attempted: 0, failed: 0, scored: 0 },
});
const runs = [run("candidate", "en", "2026-09-16T13:00:00Z"), run("baseline"), run("portuguese", "pt"), {
  ...run("interrupted"), evaluation: { status: "cancelled", retrieval_complete: false, with_answers: false, attempted: 0, failed: 0, scored: 0 },
}];
const metric = { n: 2, baseline: 0.7, candidate: 0.8, mean_delta: 0.1, ci_low: 0, ci_high: 0.2 };
const comparison: RunComparison = {
  baseline_id: "baseline", candidate_id: "candidate", compatible: true, passed: true, failures: [],
  metrics: { recall: metric }, slices: { "tag:critical": { recall: metric } }, cases: [
    { case_id: "case-1", question: "Which evidence matters?", answerable: true, tags: ["critical"], metrics: { recall: { baseline: 0.5, candidate: 0.7, delta: 0.2 } } },
    { case_id: "unknown", question: "Is there an unsupported fact?", answerable: false, tags: [], metrics: {} },
  ],
};
const json = (value: unknown) => new Response(JSON.stringify(value));
beforeEach(() => window.history.replaceState(null, "", "/compare"));
afterEach(() => vi.unstubAllGlobals());

describe("paired run comparison", () => {
  it("only offers completed candidates in the baseline scope and shows aggregate, slices and case links", async () => {
    vi.stubGlobal("fetch", async (url: string) => json(url.includes("/evals/compare?") ? comparison : runs));
    render(<ComparePage />);
    const candidates = await screen.findByRole("combobox", { name: "Candidate run" });
    expect(within(candidates).getAllByRole("option").map((option) => option.getAttribute("value"))).toEqual(["candidate"]);
    expect(within(screen.getByRole("combobox", { name: "Baseline run" })).queryByRole("option", { name: /interrupted/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Compare runs" }));
    const aggregate = await screen.findByRole("table", { name: "Aggregate retrieval comparison" });
    expect(within(aggregate).getByText("+0.100")).toBeInTheDocument();
    expect(within(aggregate).getByText("+0.000 to +0.200")).toBeInTheDocument();
    expect(screen.getByText("tag:critical")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Candidate evidence for case-1" })).toHaveAttribute("href", "/runs/candidate#case-case-1");
    expect(screen.getByText(/unanswerable.*not applicable/i)).toBeInTheDocument();
  });

  it("visibly rejects an incompatible pair supplied through a direct URL", async () => {
    window.history.replaceState(null, "", "/compare?baseline=baseline&candidate=portuguese");
    vi.stubGlobal("fetch", async (url: string) => json(url.includes("/evals/compare?") ? {
      ...comparison, candidate_id: "portuguese", compatible: false, passed: false,
      failures: ["incompatible manifest field: language"], metrics: {}, slices: {}, cases: [],
    } : runs));
    render(<ComparePage />);
    expect(await screen.findByRole("alert")).toHaveTextContent("incompatible manifest field: language");
    expect(screen.queryByRole("table", { name: "Aggregate retrieval comparison" })).not.toBeInTheDocument();
  });

  it("aborts an obsolete comparison and ignores its late response", async () => {
    const pending: { resolve: (value: Response) => void; signal: AbortSignal | null | undefined }[] = [];
    vi.stubGlobal("fetch", (url: string, init: RequestInit) => {
      if (!url.includes("/evals/compare?")) return Promise.resolve(json([...runs, run("other")]));
      return new Promise<Response>((resolve) => pending.push({ resolve, signal: init.signal }));
    });
    render(<ComparePage />);
    const candidate = await screen.findByRole("combobox", { name: "Candidate run" });
    fireEvent.click(screen.getByRole("button", { name: "Compare runs" }));
    fireEvent.change(candidate, { target: { value: "other" } });
    fireEvent.click(screen.getByRole("button", { name: "Compare runs" }));
    await act(async () => { pending[1]!.resolve(json({ ...comparison, candidate_id: "other", cases: [] })); });
    await act(async () => { pending[0]!.resolve(json(comparison)); });
    expect(pending[0]?.signal?.aborted).toBe(true);
    expect(screen.queryByText("Which evidence matters?")).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open candidate run" })).toHaveAttribute("href", "/runs/other");
  });
});
