-- Split eval latency: latency_ms now measures retrieval only; answer_ms covers
-- the LLM calls (generation + judging) so runs stay comparable across configs.
ALTER TABLE eval_case_results
  ADD COLUMN IF NOT EXISTS answer_ms int NOT NULL DEFAULT 0;
