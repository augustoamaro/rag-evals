import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { CaseResult } from "@/lib/api";
import { CaseEvidence } from "./CaseEvidence";

const failedJudgeCase: CaseResult = {
  case_id: "case-1",
  strategy: "hybrid",
  retrieved_ids: ["doc:0"],
  recall: 1,
  precision: 0.5,
  mrr: 1,
  ndcg: 0.9,
  answer: "A retained generated answer.",
  judge_scores: null,
  cost_usd: 0.025,
  latency_ms: 12,
  answer_ms: 450,
  generation: {
    status: "succeeded",
    usage: { input_tokens: 120, output_tokens: 30 },
    cost_usd: 0.02,
    latency_ms: 300,
    error_type: null,
  },
  judging: {
    status: "failed",
    usage: { input_tokens: 80, output_tokens: 0 },
    cost_usd: 0.005,
    latency_ms: 150,
    error_type: "provider_timeout",
  },
  cost_complete: false,
  question: "What is the consistency trade-off?",
  reference_answer: "Availability can be preferred during a partition.",
  retrieved_context: [
    {
      chunk: {
        id: "doc:0",
        document_id: "doc",
        ordinal: 0,
        text: "A network partition forces a consistency or availability choice.",
        token_count: 10,
      },
      score: 0.91,
      rank: 1,
    },
  ],
};

describe("CaseEvidence", () => {
  it("shows unanswerable retrieval as not applicable rather than zero quality", () => {
    render(
      <CaseEvidence result={{
        ...failedJudgeCase,
        recall: 0,
        ndcg: 0,
        mrr: 0,
        answerable: false,
        evidence_ids: [],
        covered_evidence_ids: [],
        evidence_recall: null,
      }} />,
    );

    expect(screen.getByText("Retrieval N/A")).toBeInTheDocument();
    expect(screen.getByText("Evidence N/A")).toBeInTheDocument();
    expect(screen.getByText(/excluded from retrieval-quality averages/i)).toBeInTheDocument();
    expect(screen.queryByText("Recall 0.00")).not.toBeInTheDocument();
  });

  it("shows evidence coverage, language and retained word offsets", () => {
    render(
      <CaseEvidence result={{
        ...failedJudgeCase,
        answerable: true,
        language: "pt",
        split: "dev",
        evidence_ids: ["first", "second"],
        covered_evidence_ids: ["first"],
        evidence_recall: 0.5,
        retrieved_context: [{
          rank: 1,
          score: 0.91,
          chunk: {
            id: "doc:0", document_id: "doc", ordinal: 0,
            text: "Retained span.", token_count: 2, word_start: 3, word_end: 5,
          },
        }],
      }} />,
    );

    expect(screen.getByText("Evidence 0.50")).toBeInTheDocument();
    expect(screen.getByText("1 / 2 evidence units covered")).toBeInTheDocument();
    expect(screen.getByText("dev · Portuguese")).toBeInTheDocument();
    expect(screen.getByText("doc · words 3–5 (end exclusive)")).toBeInTheDocument();
  });

  it("retains the answer and exposes stage failure, usage, cost, and evidence", () => {
    render(<CaseEvidence result={failedJudgeCase} />);

    expect(screen.getByText("A retained generated answer.")).toBeInTheDocument();
    expect(screen.getByText("Judging failed")).toBeInTheDocument();
    expect(screen.getByText("provider_timeout")).toBeInTheDocument();
    expect(screen.getByText("120 in / 30 out")).toBeInTheDocument();
    expect(screen.getByText("80 in / 0 out")).toBeInTheDocument();
    expect(screen.getByText("$0.0250 known · incomplete")).toBeInTheDocument();
    expect(screen.getByText("What is the consistency trade-off?")).toBeInTheDocument();
    expect(
      screen.getByText("Availability can be preferred during a partition."),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "A network partition forces a consistency or availability choice.",
      ),
    ).toBeInTheDocument();
  });

  it("labels evidence missing from historical rows", () => {
    render(
      <CaseEvidence
        result={{
          ...failedJudgeCase,
          case_id: "legacy-case",
          answer: "A historical retained answer.",
          generation: null,
          judging: null,
          cost_complete: null,
          question: null,
          reference_answer: null,
          retrieved_context: null,
        }}
      />,
    );

    expect(screen.getByText("Historical evidence unavailable")).toBeInTheDocument();
    expect(screen.getByText("A historical retained answer.")).toBeInTheDocument();
    expect(screen.getByText("doc:0")).toBeInTheDocument();
  });

  it("flags cited ids that match no retrieved passage", () => {
    render(<CaseEvidence result={{ ...failedJudgeCase, invalid_citations: ["doc:9", "nope"] }} />);

    expect(screen.getByText(/cites passages that were not retrieved/i)).toHaveTextContent("doc:9, nope");
  });

  it.each([[[]], [null], [undefined]])("shows no citation warning for %j", (invalid) => {
    render(<CaseEvidence result={{ ...failedJudgeCase, invalid_citations: invalid }} />);

    expect(screen.queryByText(/cites passages that were not retrieved/i)).not.toBeInTheDocument();
  });
});
