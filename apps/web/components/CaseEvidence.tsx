import type { CaseResult, StageResult } from "@/lib/api";
import { languageLabel } from "@/lib/benchmark-scope";
import { RetrievalScore } from "./RetrievalScore";

function StageReadout({ label, stage }: { label: string; stage: StageResult }) {
  const status = `${label} ${stage.status}`;

  return (
    <div className={`stage stage--${stage.status}`}>
      <div className="stage__heading">{status}</div>
      <div className="stage__facts">
        <span>{stage.latency_ms}ms</span>
        <span>
          {stage.usage
            ? `${stage.usage.input_tokens} in / ${stage.usage.output_tokens} out`
            : "Usage unavailable"}
        </span>
        <span>
          {stage.cost_usd === null ? "Cost unknown" : `$${stage.cost_usd.toFixed(4)}`}
        </span>
      </div>
      {stage.error_type ? <div className="stage__error">{stage.error_type}</div> : null}
    </div>
  );
}

function caseCost(result: CaseResult) {
  const formatted = `$${result.cost_usd.toFixed(4)}`;
  if (result.cost_complete === null) {
    return `${formatted} recorded · completeness unknown`;
  }
  if (result.cost_complete) return `${formatted} complete`;
  return `${formatted} known · incomplete`;
}

export function CaseEvidence({ result }: { result: CaseResult }) {
  const hasHistoricalEvidence =
    result.question !== null ||
    result.reference_answer !== null ||
    result.retrieved_context !== null;
  const retrievalApplicable = result.answerable !== false && result.evidence_ids?.length !== 0;
  const hasEvidenceLabels = result.evidence_ids != null || result.answerable != null;
  const scope = [result.split, result.language ? languageLabel(result.language) : null]
    .filter(Boolean).join(" · ");

  return (
    <details className="case-evidence" id={`case-${result.case_id}`}>
      <summary className="case-evidence__summary">
        <span className="case-evidence__id">{result.case_id}</span>
        <span className="case-evidence__metrics">
          {retrievalApplicable ? (
            <>
              <span>Recall {result.recall.toFixed(2)}</span>
              <span>nDCG {result.ndcg.toFixed(2)}</span>
              <span>MRR {result.mrr.toFixed(2)}</span>
            </>
          ) : <span>Retrieval N/A</span>}
          {hasEvidenceLabels ? (
            <span>
              Evidence {retrievalApplicable ? result.evidence_recall?.toFixed(2) ?? "—" : "N/A"}
            </span>
          ) : null}
        </span>
        <span className="case-evidence__hint">Inspect evidence</span>
      </summary>

      <div className="case-evidence__body">
        {scope ? <div className="case-evidence__scope">{scope}</div> : null}
        {!retrievalApplicable ? (
          <p className="case-evidence__coverage">
            {result.answerable === false ? "Unanswerable case" : "No labeled evidence"}; excluded from retrieval-quality averages.
          </p>
        ) : result.evidence_ids && result.covered_evidence_ids ? (
          <p className="case-evidence__coverage">
            {result.covered_evidence_ids.length} / {result.evidence_ids.length} evidence units covered
          </p>
        ) : null}
        <div className="case-evidence__cost">{caseCost(result)}</div>

        {result.generation || result.judging ? (
          <div className="stage-grid">
            {result.generation ? (
              <StageReadout label="Generation" stage={result.generation} />
            ) : null}
            {result.judging ? (
              <StageReadout label="Judging" stage={result.judging} />
            ) : null}
          </div>
        ) : null}

        {hasHistoricalEvidence ? (
          <div className="evidence-grid">
            <div className="evidence-copy">
              <div className="evidence-block">
                <div className="evidence-block__label">Question</div>
                <div>{result.question ?? "Not recorded"}</div>
              </div>
              <div className="evidence-block">
                <div className="evidence-block__label">Reference answer</div>
                <div>{result.reference_answer === "" ? "Not provided for this dataset" : result.reference_answer ?? "Not recorded"}</div>
              </div>
              <div className="evidence-block">
                <div className="evidence-block__label">Generated answer</div>
                <div>{result.answer ?? "No answer generated"}</div>
              </div>
            </div>

            <div className="evidence-context">
              <div className="evidence-block__label">Retrieved context</div>
              <p className="score-help">Scores use different scales for each method. Legacy records may lack score provenance.</p>
              {result.retrieved_context?.length ? (
                result.retrieved_context.map((item) => (
                  <article className="context-chunk" key={item.chunk.id}>
                    <div className="context-chunk__meta">
                      <span>
                        #{item.rank} · {item.chunk.id}
                      </span>
                    </div>
                    <RetrievalScore {...item} />
                    {item.chunk.word_start != null && item.chunk.word_end != null ? (
                      <div className="context-chunk__offsets">
                        {item.chunk.document_id} · words {item.chunk.word_start}–{item.chunk.word_end} (end exclusive)
                      </div>
                    ) : null}
                    <div>{item.chunk.text}</div>
                  </article>
                ))
              ) : (
                <div className="muted">No context retrieved.</div>
              )}
            </div>
          </div>
        ) : (
          <div className="legacy-evidence">
            <div className="legacy-evidence__title">Historical evidence unavailable</div>
            {result.answer ? (
              <div className="legacy-evidence__answer">
                <div className="evidence-block__label">Generated answer</div>
                <div>{result.answer}</div>
              </div>
            ) : null}
            <div>
              This run predates stored questions and context. Retrieved IDs: {" "}
              <span className="num">
                {result.retrieved_ids.length ? result.retrieved_ids.join(", ") : "none"}
              </span>
            </div>
          </div>
        )}
      </div>
    </details>
  );
}
