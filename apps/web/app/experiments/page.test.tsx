import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { EvaluationJob, EvaluationOptions, JobRequest } from "@/lib/api";
import ExperimentsPage from "./page";

const options: EvaluationOptions = {
  dataset: { name: "Engineering", version: "2026.1", index_id: "index-1" },
  default_language: "en", default_split: "smoke", strategies: ["hybrid", "dense", "sparse"], worker_available: true,
  selections: [
    { split: "dev", language: "pt", case_count: 3, answers_available: false, answer_unavailable_reason: "Reference answers unavailable for Portuguese" },
    { split: "smoke", language: "en", case_count: 2, answers_available: true, answer_unavailable_reason: null },
  ],
};
const job: EvaluationJob = {
  id: "job-1", run_id: null, request_id: "request", config: { strategy: "hybrid", k: 10, split: "smoke", language: "en", with_answers: false },
  status: "queued", total: 2, completed: 0, stage: "queued", case_id: null, error_type: null,
  created_at: "2026-09-16T14:00:00Z", updated_at: "2026-09-16T14:00:00Z",
};
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status });

beforeEach(() => sessionStorage.clear());
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); vi.useRealTimers(); });

describe("experiment launch", () => {
  it("defaults to English and keeps Portuguese separate with an answer availability reason", async () => {
    vi.stubGlobal("fetch", async (url: string) => json(url.endsWith("/options") ? options : []));
    render(<ExperimentsPage />);
    const language = await screen.findByRole("combobox", { name: "Language" });
    expect(language).toHaveValue("en");
    expect(screen.getByRole("combobox", { name: "Split" })).toHaveValue("smoke");
    expect(screen.getByRole("checkbox", { name: "Generate and judge answers" })).not.toBeChecked();
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.change(language, { target: { value: "pt" } });
    expect(screen.getByRole("combobox", { name: "Split" })).toHaveValue("dev");
    expect(screen.getByRole("checkbox")).toBeDisabled();
    expect(screen.getByRole("checkbox")).not.toBeChecked();
    expect(screen.getByText("Reference answers unavailable for Portuguese")).toBeInTheDocument();
  });

  it("disables launching when the worker is unavailable", async () => {
    vi.stubGlobal("fetch", async (url: string) => json(url.endsWith("/options") ? { ...options, worker_available: false } : []));
    render(<ExperimentsPage />);
    expect(await screen.findByRole("button", { name: "Launch evaluation" })).toBeDisabled();
    expect(screen.getByText(/worker is unavailable/i)).toBeInTheDocument();
  });

  it("retains options and a disabled launch when the unavailable worker cannot list jobs", async () => {
    vi.stubGlobal("fetch", async (url: string) => url.endsWith("/options")
      ? json({ ...options, dataset: null, worker_available: false })
      : json({ detail: "evaluation worker is unavailable" }, 503));
    render(<ExperimentsPage />);
    expect(await screen.findByRole("button", { name: "Launch evaluation" })).toBeDisabled();
    expect(screen.getByRole("combobox", { name: "Language" })).toHaveValue("en");
    expect(screen.getByRole("alert")).toHaveTextContent("evaluation worker is unavailable");
  });

  it("launches, polls checkpointed progress, and cancels without keeping a poll alive", async () => {
    const sent: JobRequest[] = [];
    let polls = 0;
    vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
      if (url.endsWith("/options")) return json(options);
      if (url.endsWith("/cancel")) return json({ ...job, status: "cancelled", run_id: "run-1", completed: 1, updated_at: "2026-09-16T14:00:03Z" });
      if (url.endsWith("/jobs/job-1")) {
        polls += 1;
        return json({ ...job, status: "running", run_id: "run-1", completed: 1, stage: "retrieval", case_id: "case-2", updated_at: "2026-09-16T14:00:02Z" });
      }
      if (init.method === "POST") { sent.push(JSON.parse(String(init.body)) as JobRequest); return json(job, 202); }
      return json([]);
    });
    render(<ExperimentsPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Launch evaluation" }));
    expect(await screen.findByRole("link", { name: "Open run" })).toHaveAttribute("href", "/runs/run-1");
    expect(screen.getByText("1 / 2 cases")).toBeInTheDocument();
    expect(screen.getByText(/retrieval.*case-2/i)).toBeInTheDocument();
    expect(sent[0]).toMatchObject({ language: "en", split: "smoke", with_answers: false, strategy: "hybrid", k: 10 });
    expect(sent[0]?.request_id).toMatch(/^[0-9a-f-]{36}$/);
    vi.useFakeTimers();
    fireEvent.click(screen.getByRole("button", { name: "Cancel evaluation" }));
    await act(async () => {});
    expect(screen.getByText("cancelled", { selector: ".job-status" })).toBeInTheDocument();
    const before = polls;
    await act(async () => { vi.advanceTimersByTime(5000); });
    expect(polls).toBe(before);
  });

  it.each(["network", "503"])("reuses the same request after an ambiguous %s failure and page reload", async (failure) => {
    const sent: JobRequest[] = [];
    vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
      if (url.endsWith("/options")) return json(options);
      if (init.method === "POST") {
        sent.push(JSON.parse(String(init.body)) as JobRequest);
        if (sent.length === 1) {
          if (failure === "network") throw new TypeError("Connection interrupted");
          return json({ detail: "Service temporarily unavailable" }, 503);
        }
        return json({ ...job, status: "completed", request_id: sent[1]!.request_id }, 202);
      }
      return json([]);
    });
    const first = render(<ExperimentsPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Launch evaluation" }));
    expect(await screen.findByRole("button", { name: "Retry launch" })).toBeEnabled();
    expect(screen.getByRole("combobox", { name: "Strategy" })).toBeDisabled();
    first.unmount();
    render(<ExperimentsPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Retry launch" }));
    expect(await screen.findByText("completed", { selector: ".job-status" })).toBeInTheDocument();
    expect(sent).toHaveLength(2);
    expect(sent[1]).toEqual(sent[0]);
  });

  it("recovers an active job on mount and aborts its progress request on unmount", async () => {
    let progressSignal: AbortSignal | null | undefined;
    vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
      if (url.endsWith("/options")) return json(options);
      if (url.endsWith("/jobs/job-1")) {
        progressSignal = init.signal;
        return new Promise<Response>(() => {});
      }
      return json([job]);
    });
    const view = render(<ExperimentsPage />);
    expect(await screen.findByText("queued", { selector: ".job-status" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Launch evaluation" })).toBeDisabled();
    expect(screen.queryByRole("link", { name: "Open run" })).not.toBeInTheDocument();
    await waitFor(() => expect(progressSignal).toBeDefined());
    view.unmount();
    expect(progressSignal?.aborted).toBe(true);
  });

  it("retains an unconfirmed in-memory request across refresh when storage is unavailable", async () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new DOMException("Storage unavailable", "SecurityError"); });
    const sent: JobRequest[] = [];
    vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
      if (url.endsWith("/options")) return json(options);
      if (init.method === "POST") {
        sent.push(JSON.parse(String(init.body)) as JobRequest);
        if (sent.length === 1) throw new TypeError("Connection interrupted");
        return json({ ...job, status: "completed", request_id: sent[1]!.request_id }, 202);
      }
      return json([]);
    });
    render(<ExperimentsPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Launch evaluation" }));
    expect(await screen.findByRole("button", { name: "Retry launch" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Refresh jobs" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Refresh jobs" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Retry launch" }));
    expect(await screen.findByText("completed", { selector: ".job-status" })).toBeInTheDocument();
    expect(sent).toHaveLength(2);
    expect(sent[1]).toEqual(sent[0]);
  });
});
