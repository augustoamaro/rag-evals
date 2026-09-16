import { afterEach, describe, expect, it, vi } from "vitest";
import { listRuns, query } from "./api";

afterEach(() => vi.unstubAllGlobals());

describe("HTTP boundary", () => {
  it("keeps a useful server error reason", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify({ detail: "An evaluation is already active" }), { status: 409 }));
    await expect(listRuns()).rejects.toThrow("An evaluation is already active");
  });

  it("passes cancellation through read and query requests", async () => {
    const signals: (AbortSignal | null | undefined)[] = [];
    vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
      signals.push(init.signal);
      return new Response(JSON.stringify([]));
    });
    const controller = new AbortController();
    await listRuns(controller.signal);
    await query("Question", 5, "dense", controller.signal);
    expect(signals).toEqual([controller.signal, controller.signal]);
  });
});
