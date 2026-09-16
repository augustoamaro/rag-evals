import type { EvaluationSummary } from "@/lib/api";

interface EvaluationStatusProps {
  evaluation: EvaluationSummary | null;
  costUsd: number;
  costComplete: boolean | null;
  compact?: boolean;
}

function describeEvaluation(evaluation: EvaluationSummary | null) {
  if (evaluation === null) {
    return {
      tone: "unknown",
      title: "Evaluation status unknown",
      detail: "Legacy run · coverage unavailable",
    };
  }
  if (evaluation.status === "running" || evaluation.status === "cancelled") {
    return {
      tone: "partial",
      title: evaluation.status === "running" ? "Evaluation running" : "Evaluation cancelled",
      detail: "Retrieval incomplete · excluded from comparisons",
    };
  }
  if (evaluation.retrieval_complete === false) {
    return {
      tone: "failed",
      title: "Evaluation incomplete",
      detail: `Retrieval incomplete${evaluation.error_type ? ` · ${evaluation.error_type}` : ""}`,
    };
  }
  if (!evaluation.with_answers) {
    return {
      tone: "retrieval",
      title: "Retrieval only",
      detail: "No answer judging requested",
    };
  }

  const coverage = `${evaluation.scored} of ${evaluation.attempted} scored · ${evaluation.failed} failed`;
  if (evaluation.status === "failed") {
    return { tone: "failed", title: "Answer evaluation failed", detail: coverage };
  }
  if (evaluation.status === "partial") {
    return { tone: "partial", title: "Partial answer evaluation", detail: coverage };
  }
  return { tone: "complete", title: "Answer evaluation complete", detail: coverage };
}

function describeCost(costUsd: number, costComplete: boolean | null) {
  const formatted = `$${costUsd.toFixed(4)}`;
  if (costComplete === null) return `${formatted} recorded · completeness unknown`;
  if (costComplete) return `${formatted} complete`;
  return `${formatted} known · incomplete`;
}

export function EvaluationStatus({
  evaluation,
  costUsd,
  costComplete,
  compact = false,
}: EvaluationStatusProps) {
  const description = describeEvaluation(evaluation);

  return (
    <section
      className={`evaluation-status evaluation-status--${description.tone}${
        compact ? " evaluation-status--compact" : ""
      }`}
      aria-label="Evaluation reliability"
    >
      <div className="evaluation-status__signal" aria-hidden="true" />
      <div>
        <div className="evaluation-status__title">{description.title}</div>
        <div className="evaluation-status__detail">{description.detail}</div>
        <div className="evaluation-status__cost">
          {describeCost(costUsd, costComplete)}
        </div>
      </div>
    </section>
  );
}
