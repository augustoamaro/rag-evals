import { describe, expect, it } from "vitest";
import type { RunSummary } from "./api";
import { benchmarkScope, groupBenchmarkRuns, languageLabel } from "./benchmark-scope";

const manifest: Record<string, unknown> = {
  schema_version: 2,
  dataset_name: "rag-engineering",
  dataset_version: "2026-09-16.1",
  dataset_sha256: "a".repeat(64),
  case_set_sha256: "b".repeat(64),
  split: "smoke",
  language: "en",
  k: 10,
  retrieval_level: "chunk",
  evidence_metric_version: "document-or-word-union-v1",
  ranking_metric_version: "span-overlap-chunks-v1",
  strategy: "dense",
  index_id: "idx-original",
};

function run(
  id: string,
  config: Record<string, unknown> = {},
  createdAt = "2026-09-16T12:00:00Z",
): RunSummary {
  return {
    id,
    created_at: createdAt,
    config: { ...manifest, ...config },
    retrieval_metrics: { recall: 1, precision: 0.5, mrr: 1, ndcg: 0.9 },
    answer_metrics: null,
    cost_usd: 0,
    latency_p50_ms: 10,
    latency_p95_ms: 15,
    git_sha: null,
    evaluation: null,
    cost_complete: true,
  };
}

describe("benchmark compatibility", () => {
  it.each(["running", "cancelled", "failed"] as const)("keeps a %s retrieval prefix out of comparative history", (status) => {
    const prefix: RunSummary = { ...run("prefix"), evaluation: {
      status, with_answers: false, attempted: 0, scored: 0, failed: 0,
      retrieval_complete: false,
    } };
    const result = groupBenchmarkRuns([prefix, run("complete")]);
    expect(result.groups[0]?.runs.map((item) => item.id)).toEqual(["complete"]);
    expect(result.unscoped.map((item) => item.id)).toEqual(["prefix"]);
  });

  it("excludes a cancelled run even if its retrieval flag is inconsistent", () => {
    const result = groupBenchmarkRuns([{ ...run("cancelled"), evaluation: {
      status: "cancelled", with_answers: false, attempted: 0, scored: 0, failed: 0,
      retrieval_complete: true,
    } }]);
    expect(result.groups).toEqual([]);
  });

  it("retains full retrieval from a partially judged evaluation", () => {
    const result = groupBenchmarkRuns([{ ...run("partial"), evaluation: {
      status: "partial", with_answers: true, attempted: 2, scored: 1, failed: 1,
      retrieval_complete: true,
    } }]);
    expect(result.groups[0]?.runs[0]?.id).toBe("partial");
  });

  it("excludes a run with an interruption error even if its completion flag says true", () => {
    const result = groupBenchmarkRuns([{ ...run("interrupted"), evaluation: {
      status: "failed", with_answers: true, attempted: 2, scored: 0, failed: 0,
      retrieval_complete: true, error_type: "worker_interrupted",
    } }]);
    expect(result.groups).toEqual([]);
    expect(result.unscoped[0]?.id).toBe("interrupted");
  });

  it("retains language codes that do not have a display name", () => {
    expect(languageLabel("constructor")).toBe("constructor");
    expect(languageLabel("es")).toBe("es");
  });

  it("compares matching labels across controlled model, index and strategy changes", () => {
    const baseline = run("baseline");
    const candidate = run("candidate", {
      strategy: "hybrid",
      index_id: "idx-candidate",
      embedding_identity: { model: "another-model" },
    });

    const result = groupBenchmarkRuns([baseline, candidate]);

    expect(result.groups).toHaveLength(1);
    expect(result.groups[0]?.runs.map((item) => item.id).sort()).toEqual([
      "baseline", "candidate",
    ]);
    expect(result.unscoped).toEqual([]);
  });

  it.each([
    ["dataset_sha256", "c".repeat(64)],
    ["case_set_sha256", "c".repeat(64)],
    ["split", "dev"],
    ["language", "pt"],
    ["k", 5],
    ["retrieval_level", "document"],
    ["evidence_metric_version", "word-union-v2"],
    ["ranking_metric_version", "document-first-hit-v1"],
  ])("separates runs when %s differs", (field, value) => {
    const result = groupBenchmarkRuns([run("baseline"), run("changed", { [field]: value })]);

    expect(result.groups).toHaveLength(2);
    expect(result.groups.every((group) => group.runs.length === 1)).toBe(true);
  });

  it.each([
    ["schema_version", 1],
    ["schema_version", "2"],
    ["schema_version", 3],
    ["dataset_sha256", "short-hash"],
    ["case_set_sha256", undefined],
    ["split", ""],
    ["split", "train"],
    ["language", null],
    ["language", " "],
    ["k", "10"],
    ["k", 0],
    ["k", 1.5],
    ["k", Number.NaN],
    ["retrieval_level", "unknown"],
    ["evidence_metric_version", null],
    ["ranking_metric_version", ""],
  ])("keeps invalid %s metadata out of comparative history", (field, value) => {
    const invalid = run("invalid", { [field]: value });

    expect(benchmarkScope(invalid.config)).toBeNull();
    const result = groupBenchmarkRuns([invalid]);
    expect(result.groups).toEqual([]);
    expect(result.unscoped.map((item) => item.id)).toEqual(["invalid"]);
    expect(result.defaultKey).toBeNull();
  });

  it("retains legacy runs for individual inspection without inferring a scope", () => {
    const legacy = { ...run("legacy"), config: { strategy: "hybrid", k: 10 } };
    const result = groupBenchmarkRuns([run("modern"), legacy]);

    expect(result.groups[0]?.runs.map((item) => item.id)).toEqual(["modern"]);
    expect(result.unscoped.map((item) => item.id)).toEqual(["legacy"]);
  });

  it("defaults to the newest English scope even when Portuguese is more recent", () => {
    const oldEnglish = run("old-en", { k: 5 }, "2026-09-14T12:00:00Z");
    const newEnglish = run("new-en", {}, "2026-09-15T12:00:00Z");
    const portuguese = run("pt", { language: "pt" }, "2026-09-16T12:00:00Z");
    const result = groupBenchmarkRuns([oldEnglish, newEnglish, portuguese]);

    expect(result.groups.find((group) => group.key === result.defaultKey)?.runs[0]?.id)
      .toBe("new-en");
    expect(result.groups[0]?.runs[0]?.id).toBe("pt");
  });

  it("sorts each scope newest first and falls back to the newest available language", () => {
    const older = run("older", { language: "pt" }, "2026-09-14T12:00:00Z");
    const newer = run("newer", { language: "pt" }, "2026-09-16T12:00:00Z");
    const result = groupBenchmarkRuns([older, newer]);

    expect(result.groups[0]?.runs.map((item) => item.id)).toEqual(["newer", "older"]);
    expect(result.defaultKey).toBe(result.groups[0]?.key);
  });
});
