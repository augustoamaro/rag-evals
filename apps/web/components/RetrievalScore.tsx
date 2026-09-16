import type { ScoreProvenance } from "@/lib/api";

const names: Record<string, string> = {
  cosine_similarity: "Cosine similarity", ts_rank_cd: "Text rank", rrf: "RRF", cross_encoder: "Cross-encoder",
};

export function RetrievalScore({ score, score_type, score_components }: ScoreProvenance) {
  const label = score_type && Object.hasOwn(names, score_type) ? names[score_type] : score_type;
  return (
    <div className="retrieval-score">
      <span>{score_type === "rank_only" ? "Rank only · no relevance score"
        : !score_type ? `${score?.toFixed(4) ?? "No score"} · score provenance unavailable`
          : `${label} · ${score?.toFixed(4) ?? "not available"}`}</span>
      {score_components && Object.keys(score_components).length ? (
        <span className="retrieval-score__components">
          {Object.entries(score_components).map(([name, value]) => <span key={name}>{name}: {value.toFixed(4)}</span>)}
        </span>
      ) : null}
    </div>
  );
}
