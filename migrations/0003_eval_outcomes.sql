-- Null means unknown for historical runs; never invent successful coverage.
ALTER TABLE eval_runs ADD COLUMN evaluation jsonb;
ALTER TABLE eval_runs ADD COLUMN cost_complete boolean;

ALTER TABLE eval_case_results ADD COLUMN generation jsonb;
ALTER TABLE eval_case_results ADD COLUMN judging jsonb;
ALTER TABLE eval_case_results ADD COLUMN cost_complete boolean;
ALTER TABLE eval_case_results ADD COLUMN question text;
ALTER TABLE eval_case_results ADD COLUMN reference_answer text;
ALTER TABLE eval_case_results ADD COLUMN retrieved_context jsonb;
