import type { RunSummary } from "./api";

export interface BenchmarkScope {
  key: string;
  datasetName: string;
  datasetVersion: string;
  datasetSha256: string;
  caseSetSha256: string;
  split: "smoke" | "dev" | "test";
  language: string;
  k: number;
  retrievalLevel: "chunk" | "document";
  evidenceMetricVersion: string;
  rankingMetricVersion: string;
}

export interface BenchmarkGroup extends BenchmarkScope {
  runs: RunSummary[];
}

const sha256 = /^[a-f0-9]{64}$/;
const nonempty = (value: unknown): value is string =>
  typeof value === "string" && value.trim().length > 0;

export function benchmarkScope(config: Record<string, unknown>): BenchmarkScope | null {
  const {
    schema_version: schema,
    dataset_sha256: datasetSha256,
    case_set_sha256: caseSetSha256,
    split,
    language,
    k,
    retrieval_level: retrievalLevel,
    evidence_metric_version: evidenceMetricVersion,
    ranking_metric_version: rankingMetricVersion,
  } = config;
  if (
    schema !== 2 ||
    !nonempty(datasetSha256) || !sha256.test(datasetSha256) ||
    !nonempty(caseSetSha256) || !sha256.test(caseSetSha256) ||
    (split !== "smoke" && split !== "dev" && split !== "test") ||
    !nonempty(language) ||
    typeof k !== "number" || !Number.isInteger(k) || k < 1 || k > 50 ||
    (retrievalLevel !== "chunk" && retrievalLevel !== "document") ||
    !nonempty(evidenceMetricVersion) || !nonempty(rankingMetricVersion)
  ) {
    return null;
  }
  return {
    key: JSON.stringify([
      schema, datasetSha256, caseSetSha256, split, language, k,
      retrievalLevel, evidenceMetricVersion, rankingMetricVersion,
    ]),
    datasetName: nonempty(config.dataset_name) ? config.dataset_name : "Unnamed dataset",
    datasetVersion: nonempty(config.dataset_version) ? config.dataset_version : "unrecorded",
    datasetSha256,
    caseSetSha256,
    split,
    language,
    k,
    retrievalLevel,
    evidenceMetricVersion,
    rankingMetricVersion,
  };
}

export function languageLabel(language: string): string {
  if (language === "en") return "English";
  if (language === "pt") return "Portuguese";
  return language;
}

export function benchmarkLabel(scope: BenchmarkScope): string {
  return `${scope.datasetName} · ${scope.datasetVersion} · ${scope.split} · ${languageLabel(scope.language)} · k=${scope.k} · ${scope.retrievalLevel} retrieval`;
}

export function retrievalIsComplete(run: RunSummary): boolean {
  return run.evaluation?.retrieval_complete !== false
    && run.evaluation?.status !== "running"
    && run.evaluation?.status !== "cancelled"
    && run.evaluation?.error_type == null;
}

export function groupBenchmarkRuns(runs: RunSummary[]): {
  groups: BenchmarkGroup[];
  unscoped: RunSummary[];
  defaultKey: string | null;
} {
  const groups = new Map<string, BenchmarkGroup>();
  const unscoped: RunSummary[] = [];
  const ordered = [...runs].sort((a, b) =>
    (Date.parse(b.created_at) || 0) - (Date.parse(a.created_at) || 0) || a.id.localeCompare(b.id),
  );
  for (const run of ordered) {
    const scope = benchmarkScope(run.config);
    if (!scope || !retrievalIsComplete(run)) {
      unscoped.push(run);
      continue;
    }
    const group = groups.get(scope.key);
    if (group) group.runs.push(run);
    else groups.set(scope.key, { ...scope, runs: [run] });
  }
  const values = [...groups.values()];
  return {
    groups: values,
    unscoped,
    defaultKey: values.find((group) => group.language === "en")?.key ?? values[0]?.key ?? null,
  };
}
