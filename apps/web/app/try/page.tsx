"use client";

import { useState } from "react";
import { query, type QueryResponse } from "@/lib/api";

const STRATEGIES = ["hybrid", "dense", "sparse", "hybrid_rerank"];

export default function TryPage() {
  const [question, setQuestion] = useState("What does the CAP theorem force you to trade off?");
  const [strategy, setStrategy] = useState("hybrid");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      setResult(await query(question, 5, strategy));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <h1 className="page-title">Query playground</h1>
      <p className="page-sub">
        Ask a question against the indexed corpus and inspect what hybrid retrieval returns.
      </p>

      <form
        className="query-form"
        onSubmit={(e) => {
          e.preventDefault();
          void run();
        }}
      >
        <input
          className="query-input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          aria-label="Question"
        />
        <select
          className="select"
          value={strategy}
          onChange={(e) => setStrategy(e.target.value)}
          aria-label="Strategy"
        >
          {STRATEGIES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <button className="btn" type="submit" disabled={loading}>
          {loading ? "Retrieving…" : "Query"}
        </button>
      </form>

      {error ? <div className="empty">Could not reach the API.</div> : null}

      {result ? (
        <>
          <div className="section-title">Answer</div>
          <div className="answer">
            {result.answer}
            {result.citations.length > 0 ? (
              <div className="metric__sub" style={{ marginTop: 10 }}>
                cited: {result.citations.map((c) => c.chunk_id).join(", ")}
              </div>
            ) : null}
          </div>

          <div className="section-title">Retrieved chunks</div>
          {result.chunks.map((c) => (
            <div className="chunk" key={c.id}>
              <div className="chunk__meta">
                <span className="chunk__id">[{c.id}]</span>
                <span>
                  rank {c.rank} · score {c.score.toFixed(3)}
                </span>
              </div>
              <div className="chunk__text">{c.text}</div>
            </div>
          ))}
        </>
      ) : null}
    </>
  );
}
