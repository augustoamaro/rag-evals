import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RetrievalScore } from "./RetrievalScore";

describe("retrieval score provenance", () => {
  it("shows the RRF total and available components without comparing scales", () => {
    render(<RetrievalScore score={0.032} score_type="rrf" score_components={{ dense: 0.84, sparse: 0.22 }} />);
    expect(screen.getByText("RRF · 0.0320")).toBeInTheDocument();
    expect(screen.getByText("dense: 0.8400")).toBeInTheDocument();
    expect(screen.getByText("sparse: 0.2200")).toBeInTheDocument();
  });

  it("does not invent a score for a rank-only reranker", () => {
    render(<RetrievalScore score={null} score_type="rank_only" score_components={{}} />);
    expect(screen.getByText("Rank only · no relevance score")).toBeInTheDocument();
    expect(screen.queryByText(/0\.000/)).not.toBeInTheDocument();
  });

  it("identifies records without score provenance", () => {
    render(<RetrievalScore score={0.91} />);
    expect(screen.getByText(/score provenance unavailable/i)).toBeInTheDocument();
  });
});
