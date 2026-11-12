-- Cited ids that match no retrieved passage. Null means the answer was never
-- checked (no answer generated, or a run that predates this column); an empty
-- array means every citation resolved.
ALTER TABLE eval_case_results ADD COLUMN invalid_citations jsonb;
