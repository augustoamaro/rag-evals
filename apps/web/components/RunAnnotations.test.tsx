import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RunAnnotations } from "./RunAnnotations";

const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status });
const cases = [{ case_id: "case-1", question: "Which evidence matters?" }];
afterEach(() => vi.unstubAllGlobals());

describe("run notes", () => {
  it("preserves an unsaved draft after a revision conflict and requires review before overwriting", async () => {
    let revision = 1;
    let stored = "Original note";
    const sent: { case_id: string | null; note: string; revision: number }[] = [];
    vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
      if (init.method === "PUT") {
        const body = JSON.parse(String(init.body)) as typeof sent[number];
        sent.push(body);
        if (body.revision === 1) { revision = 2; stored = "A newer saved note"; return json({ detail: "Annotation revision conflict" }, 409); }
        stored = body.note;
        revision += 1;
        return json({ case_id: null, note: stored, revision, updated_at: "2026-09-16T14:00:00Z" });
      }
      return json([{ case_id: null, note: stored, revision, updated_at: "2026-09-16T13:00:00Z" }]);
    });
    render(<RunAnnotations runId="run-1" cases={cases} />);
    const textbox = await screen.findByRole("textbox", { name: "Note" });
    await waitFor(() => expect(textbox).toHaveValue("Original note"));
    fireEvent.change(textbox, { target: { value: "My unsaved finding" } });
    fireEvent.click(screen.getByRole("button", { name: "Save note" }));
    expect(await screen.findByText("A newer saved note")).toBeInTheDocument();
    expect(textbox).toHaveValue("My unsaved finding");
    expect(screen.getByRole("button", { name: "Save note" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Save my draft over latest" }));
    expect(await screen.findByText(/note saved/i)).toBeInTheDocument();
    expect(sent).toEqual([
      { case_id: null, note: "My unsaved finding", revision: 1 },
      { case_id: null, note: "My unsaved finding", revision: 2 },
    ]);
  });

  it("saves case notes with their case id and preserves run drafts when switching scope", async () => {
    const sent: unknown[] = [];
    vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
      if (init.method === "PUT") {
        const body = JSON.parse(String(init.body)) as object;
        sent.push(body);
        return json({ ...body, revision: 1, updated_at: "2026-09-16T14:00:00Z" });
      }
      return json([]);
    });
    render(<RunAnnotations runId="run-1" cases={cases} />);
    const textbox = await screen.findByRole("textbox", { name: "Note" });
    fireEvent.change(textbox, { target: { value: "Unsaved run note" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Note scope" }), { target: { value: "case-1" } });
    fireEvent.change(textbox, { target: { value: "<b>Case finding</b>" } });
    fireEvent.click(screen.getByRole("button", { name: "Save note" }));
    expect(await screen.findByText(/note saved/i)).toBeInTheDocument();
    expect(sent).toEqual([{ case_id: "case-1", note: "<b>Case finding</b>", revision: 0 }]);
    fireEvent.change(screen.getByRole("combobox", { name: "Note scope" }), { target: { value: "" } });
    expect(textbox).toHaveValue("Unsaved run note");
  });

  it("keeps the draft editable after a network failure", async () => {
    vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
      if (init.method === "PUT") throw new TypeError("Disconnected");
      return json([]);
    });
    render(<RunAnnotations runId="run-1" cases={cases} />);
    const textbox = await screen.findByRole("textbox", { name: "Note" });
    fireEvent.change(textbox, { target: { value: "Keep this finding" } });
    fireEvent.click(screen.getByRole("button", { name: "Save note" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Disconnected");
    expect(textbox).toHaveValue("Keep this finding");
    expect(screen.getByRole("button", { name: "Save note" })).toBeEnabled();
  });
});
