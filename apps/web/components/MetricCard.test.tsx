import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MetricCard } from "./MetricCard";
import { StrategyBadge } from "./StrategyBadge";

describe("MetricCard", () => {
  it("renders label, value, and optional sub", () => {
    render(<MetricCard label="Recall@k" value="0.988" sub="k=10" />);
    expect(screen.getByText("Recall@k")).toBeInTheDocument();
    expect(screen.getByText("0.988")).toBeInTheDocument();
    expect(screen.getByText("k=10")).toBeInTheDocument();
  });
});

describe("StrategyBadge", () => {
  it("marks hybrid strategies with the accent class", () => {
    const { container } = render(<StrategyBadge strategy="hybrid" />);
    expect(container.querySelector(".badge--hybrid")).not.toBeNull();
    expect(screen.getByText("hybrid")).toBeInTheDocument();
  });
});
