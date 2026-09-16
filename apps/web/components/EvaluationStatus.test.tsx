import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EvaluationStatus } from "./EvaluationStatus";

describe("EvaluationStatus", () => {
  it.each(["running", "cancelled"] as const)("shows %s retrieval as incomplete", (status) => {
    render(<EvaluationStatus evaluation={{
      status, with_answers: false, attempted: 0, scored: 0, failed: 0,
      retrieval_complete: false,
    }} costUsd={0} costComplete />);
    expect(screen.getByText(/retrieval incomplete/i)).toBeInTheDocument();
    expect(screen.queryByText("Retrieval only")).not.toBeInTheDocument();
    expect(screen.getByText(status === "running" ? "Evaluation running" : "Evaluation cancelled")).toBeInTheDocument();
  });

  it("makes partial judged coverage and incomplete cost explicit", () => {
    render(
      <EvaluationStatus
        evaluation={{
          status: "partial",
          with_answers: true,
          attempted: 3,
          scored: 1,
          failed: 2,
        }}
        costUsd={0.03}
        costComplete={false}
      />,
    );

    expect(screen.getByText("Partial answer evaluation")).toBeInTheDocument();
    expect(screen.getByText("1 of 3 scored · 2 failed")).toBeInTheDocument();
    expect(screen.getByText("$0.0300 known · incomplete")).toBeInTheDocument();
  });

  it("does not present an all-failed judged run as complete", () => {
    render(
      <EvaluationStatus
        evaluation={{
          status: "failed",
          with_answers: true,
          attempted: 4,
          scored: 0,
          failed: 4,
        }}
        costUsd={0.0125}
        costComplete={false}
      />,
    );

    expect(screen.getByText("Answer evaluation failed")).toBeInTheDocument();
    expect(screen.getByText("0 of 4 scored · 4 failed")).toBeInTheDocument();
    expect(screen.queryByText("Answer evaluation complete")).not.toBeInTheDocument();
  });

  it("distinguishes retrieval-only runs", () => {
    render(
      <EvaluationStatus
        evaluation={{
          status: "completed",
          with_answers: false,
          attempted: 0,
          scored: 0,
          failed: 0,
        }}
        costUsd={0}
        costComplete
      />,
    );

    expect(screen.getByText("Retrieval only")).toBeInTheDocument();
    expect(screen.getByText("No answer judging requested")).toBeInTheDocument();
    expect(screen.getByText("$0.0000 complete")).toBeInTheDocument();
  });

  it("marks historical runs and cost completeness as unknown", () => {
    render(
      <EvaluationStatus evaluation={null} costUsd={0.004} costComplete={null} />,
    );

    expect(screen.getByText("Evaluation status unknown")).toBeInTheDocument();
    expect(screen.getByText("Legacy run · coverage unavailable")).toBeInTheDocument();
    expect(screen.getByText("$0.0040 recorded · completeness unknown")).toBeInTheDocument();
  });
});
