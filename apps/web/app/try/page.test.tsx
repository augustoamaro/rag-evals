import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import TryPage from "./page";

afterEach(() => vi.unstubAllGlobals());

describe("query requests", () => {
  it("keeps the submitted question and ignores a late response to an older query", async () => {
    const requests: { resolve: (value: Response) => void; signal: AbortSignal | null | undefined }[] = [];
    vi.stubGlobal("fetch", (_url: string, init: RequestInit) => new Promise<Response>((resolve) => {
      requests.push({ resolve, signal: init.signal });
    }));
    render(<TryPage />);
    fireEvent.change(screen.getByRole("textbox", { name: "Question" }), { target: { value: "First question?" } });
    fireEvent.click(screen.getByRole("button", { name: "Query" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Question" }), { target: { value: "Second question?" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Strategy" }), { target: { value: "dense" } });
    fireEvent.submit(screen.getByRole("textbox").closest("form")!);
    await act(async () => { requests[1]!.resolve(new Response(JSON.stringify({ answer: "Second answer", chunks: [], citations: [] }))); });
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Unsaved edit?" } });
    expect(screen.getByText("Second question?")).toBeInTheDocument();
    expect(screen.getByText("dense", { selector: ".badge" })).toBeInTheDocument();
    await act(async () => { requests[0]!.resolve(new Response(JSON.stringify({ answer: "First answer", chunks: [], citations: [] }))); });
    expect(screen.getByText("Second answer")).toBeInTheDocument();
    expect(screen.queryByText("First answer")).not.toBeInTheDocument();
    expect(requests[0]?.signal?.aborted).toBe(true);
  });

  it("keeps the retrieved chunks and explains a failed answer generation", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify({
      answer: "", citations: [], generation_error: "RateLimitError",
      chunks: [{ id: "cap:0", text: "CAP forces a trade-off.", score: 0.9, rank: 1, score_type: "rrf", score_components: {} }],
    })));
    render(<TryPage />);
    fireEvent.click(screen.getByRole("button", { name: "Query" }));
    const warning = await screen.findByRole("alert");
    expect(warning).toHaveTextContent("Answer generation failed (RateLimitError)");
    expect(screen.getByText("CAP forces a trade-off.")).toBeInTheDocument();
  });
});
