"use client";

import { useEffect, useRef, useState } from "react";
import { query, type QueryResponse } from "@/lib/api";
import { RetrievalScore } from "@/components/RetrievalScore";
import { StrategyBadge } from "@/components/StrategyBadge";

const STRATEGIES = ["hybrid", "dense", "sparse", "hybrid_rerank"];

export default function TryPage() {
  const [question, setQuestion] = useState("What does the CAP theorem force you to trade off?");
  const [strategy, setStrategy] = useState("hybrid");
  const [result, setResult] = useState<{ response: QueryResponse; question: string; strategy: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const request = useRef<AbortController | null>(null);

  useEffect(() => () => request.current?.abort(), []);

  const run = async () => {
    if (!question.trim()) return;
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    const submitted = { question: question.trim(), strategy };
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await query(submitted.question, 5, submitted.strategy, controller.signal);
      if (request.current === controller && !controller.signal.aborted) setResult({ response, ...submitted });
    } catch (e) {
      if (request.current === controller && !controller.signal.aborted) setError(String(e));
    } finally {
      if (request.current === controller && !controller.signal.aborted) setLoading(false);
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
        <button className="btn" type="submit" disabled={!question.trim()}>
          {loading ? "Run updated query" : "Query"}
        </button>
      </form>

      {loading ? <p role="status" className="benchmark-notice">Retrieving…</p> : null}
      {error ? <div className="workflow-error" role="alert">{error}</div> : null}

      {result ? (
        <>
          <div className="submitted-query"><p>{result.question}</p><StrategyBadge strategy={result.strategy} /></div>
          <div className="section-title">Answer</div>
          {result.response.generation_error ? (
            <p className="workflow-warning" role="alert">
              Answer generation failed ({result.response.generation_error}). The retrieved chunks below are unaffected.
            </p>
          ) : (
            <div className="answer">
              {result.response.answer}
              {result.response.citations.length > 0 ? (
                <div className="metric__sub" style={{ marginTop: 10 }}>
                  cited: {result.response.citations.map((c) => c.chunk_id).join(", ")}
                </div>
              ) : null}
            </div>
          )}

          <div className="section-title">Retrieved chunks</div>
          <p className="benchmark-notice">Scores use different scales for each retrieval method; compare ranks within a result, not scores across methods.</p>
          {result.response.chunks.map((c) => (
            <div className="chunk" key={c.id}>
              <div className="chunk__meta">
                <span className="chunk__id">[{c.id}]</span>
                <span>rank {c.rank}</span>
              </div>
              <RetrievalScore {...c} />
              <div className="chunk__text">{c.text}</div>
            </div>
          ))}
        </>
      ) : null}
    </>
  );
}
