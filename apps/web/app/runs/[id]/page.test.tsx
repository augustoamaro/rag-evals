import { Suspense } from "react";
import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { RunDetail } from "@/lib/api";
import RunPage from "./page";

const detail = (id: string): RunDetail => ({
  id, created_at: "2026-09-16T14:00:00Z", config: { strategy: "hybrid" },
  retrieval_metrics: { recall: 1, precision: 0.5, mrr: 1, ndcg: 0.9 }, answer_metrics: null,
  cost_usd: 0, latency_p50_ms: 10, latency_p95_ms: 20, git_sha: null, evaluation: null, cost_complete: true, cases: [],
});
const view = (id: string) => <Suspense fallback={<p>Loading route</p>}><RunPage params={Promise.resolve({ id })} /></Suspense>;
afterEach(() => vi.unstubAllGlobals());

describe("run navigation", () => {
  it("clears the old run immediately and aborts a request that belongs to another route", async () => {
    const pending = new Map<string, { resolve: (value: Response) => void; signal: AbortSignal | null | undefined }>();
    vi.stubGlobal("fetch", (url: string, init: RequestInit) => {
      if (url.endsWith("/annotations")) return Promise.resolve(new Response("[]"));
      return new Promise<Response>((resolve) => pending.set(url.split("/").at(-1)!, { resolve, signal: init.signal }));
    });
    let rendered: ReturnType<typeof render>;
    await act(async () => { rendered = render(view("first")); });
    await act(async () => { pending.get("first")!.resolve(new Response(JSON.stringify(detail("first")))); });
    expect(screen.getByRole("heading", { name: /Run first/ })).toBeInTheDocument();
    await act(async () => { rendered.rerender(view("second")); });
    expect(screen.queryByRole("heading", { name: /Run first/ })).not.toBeInTheDocument();
    await act(async () => { rendered.rerender(view("third")); });
    expect(pending.get("second")?.signal?.aborted).toBe(true);
    await act(async () => { pending.get("third")!.resolve(new Response(JSON.stringify(detail("third")))); });
    await act(async () => { pending.get("second")!.resolve(new Response(JSON.stringify(detail("second")))); });
    expect(screen.getByRole("heading", { name: /Run third/ })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: /Run second/ })).not.toBeInTheDocument();
  });
});
