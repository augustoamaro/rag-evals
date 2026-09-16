"use client";

import { useEffect, useId, useRef, useState } from "react";
import { ApiError, getAnnotations, saveAnnotation, type RunAnnotation } from "@/lib/api";

interface Props { runId: string; cases: { case_id: string; question: string | null }[] }
interface Draft { text: string; saved: string; revision: number; conflict: boolean; latest: RunAnnotation | null }
const emptyDraft: Draft = { text: "", saved: "", revision: 0, conflict: false, latest: null };

export function RunAnnotations(props: Props) {
  return <NotesEditor key={props.runId} {...props} />;
}

function NotesEditor({ runId, cases }: Props) {
  const [drafts, setDrafts] = useState(new Map<string, Draft>());
  const [scope, setScope] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const mutation = useRef<AbortController | null>(null);
  const inputId = useId();
  const draft = drafts.get(scope) ?? emptyDraft;

  const update = (key: string, change: (value: Draft) => Draft) => setDrafts((current) => {
    const next = new Map(current);
    next.set(key, change(current.get(key) ?? emptyDraft));
    return next;
  });

  useEffect(() => () => mutation.current?.abort(), []);
  useEffect(() => {
    const controller = new AbortController();
    setError(null);
    void getAnnotations(runId, controller.signal).then((annotations) => {
      if (controller.signal.aborted) return;
      setDrafts(new Map(annotations.map((note) => [note.case_id ?? "", {
        text: note.note, saved: note.note, revision: note.revision, conflict: false, latest: null,
      }])));
      setLoaded(true);
    }).catch((reason: unknown) => { if (!controller.signal.aborted) setError(String(reason)); });
    return () => controller.abort();
  }, [runId, reload]);

  const readConflict = async (key: string, signal: AbortSignal) => {
    const annotations = await getAnnotations(runId, signal);
    if (signal.aborted) return;
    const latest = annotations.find((item) => (item.case_id ?? "") === key)
      ?? { case_id: key || null, note: "", revision: 0, updated_at: "" };
    update(key, (value) => ({ ...value, conflict: true, latest }));
  };

  const save = async (revision = draft.revision) => {
    if (mutation.current) return;
    const key = scope;
    const controller = new AbortController();
    mutation.current = controller;
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      const saved = await saveAnnotation(runId, { case_id: key || null, note: draft.text, revision }, controller.signal);
      if (controller.signal.aborted) return;
      update(key, () => ({ text: saved.note, saved: saved.note, revision: saved.revision, conflict: false, latest: null }));
      setMessage(`Note saved · revision ${saved.revision}`);
    } catch (reason) {
      if (controller.signal.aborted) return;
      if (reason instanceof ApiError && reason.status === 409) {
        update(key, (value) => ({ ...value, conflict: true, latest: null }));
        setError("This note changed since you loaded it. Your draft is retained. Review the saved version before replacing it.");
        try { await readConflict(key, controller.signal); }
        catch (latestError) { if (!controller.signal.aborted) setError(`Your draft is retained, but the latest note could not be loaded. ${String(latestError)}`); }
      } else setError(`Note not confirmed saved. Your draft is retained. ${String(reason)}`);
    } finally {
      if (!controller.signal.aborted) setSaving(false);
      if (mutation.current === controller) mutation.current = null;
    }
  };

  const reloadConflict = async () => {
    if (mutation.current) return;
    const controller = new AbortController();
    mutation.current = controller;
    setSaving(true);
    try { await readConflict(scope, controller.signal); }
    catch (reason) { if (!controller.signal.aborted) setError(String(reason)); }
    finally {
      if (!controller.signal.aborted) setSaving(false);
      if (mutation.current === controller) mutation.current = null;
    }
  };

  return (
    <section className="notes-panel" id="notes" aria-label="Review notes">
      <h2>Review notes</h2>
      <p className="benchmark-notice">Record findings for this run or a case. Notes do not change benchmark labels.</p>
      {error ? <p className="workflow-error" role="alert">{error}</p> : null}
      {!loaded ? (
        error ? <button className="btn btn--secondary" onClick={() => setReload((value) => value + 1)}>Retry loading notes</button>
          : <p role="status" className="benchmark-notice">Loading notes…</p>
      ) : (
        <>
          <label className="note-scope">Note scope<select className="select" value={scope} disabled={saving}
            onChange={(event) => { setScope(event.target.value); setError(null); setMessage(null); }}>
            <option value="">Whole run</option>
            {cases.map((result) => <option key={result.case_id} value={result.case_id}>{result.case_id}{result.question ? ` · ${result.question}` : ""}</option>)}
          </select></label>
          <label htmlFor={inputId} className="note-label">Note</label>
          <textarea id={inputId} className="note-input" maxLength={4000} rows={4} disabled={saving}
            value={draft.text} onChange={(event) => { const text = event.target.value; update(scope, (value) => ({ ...value, text })); setMessage(null); }} />
          <div className="note-meta"><span>{draft.text.length} / 4000 characters</span><span>{draft.text !== draft.saved ? "Unsaved changes" : `Revision ${draft.revision}`}</span></div>
          {draft.conflict ? (
            <div className="note-conflict">
              <h3>Saved version</h3>
              {draft.latest ? (
                <>
                  <p className="note-saved-text">{draft.latest.note || "This note is empty."}</p>
                  <p className="benchmark-notice">Revision {draft.latest.revision}</p>
                  <div className="workflow-actions">
                    <button className="btn btn--secondary" disabled={saving} onClick={() => {
                      const latest = draft.latest!;
                      update(scope, () => ({ text: latest.note, saved: latest.note, revision: latest.revision, conflict: false, latest: null }));
                      setError(null);
                    }}>Use saved note</button>
                    <button className="btn btn--secondary" disabled={saving} onClick={() => void save(draft.latest!.revision)}>Save my draft over latest</button>
                  </div>
                </>
              ) : <button className="btn btn--secondary" disabled={saving} onClick={() => void reloadConflict()}>Reload latest note</button>}
            </div>
          ) : null}
          <div className="workflow-actions">
            <button className="btn" disabled={saving || draft.conflict || draft.text === draft.saved} onClick={() => void save()}>{saving ? "Saving…" : "Save note"}</button>
            {message ? <span className="note-success" role="status">{message}</span> : null}
          </div>
        </>
      )}
    </section>
  );
}
