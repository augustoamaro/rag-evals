-- Historical chunks and runs have no recorded evidence provenance. Leave the
-- new fields null instead of manufacturing spans, labels, or quality scores.
ALTER TABLE chunks ADD COLUMN word_start int;
ALTER TABLE chunks ADD COLUMN word_end int;

ALTER TABLE eval_runs ADD COLUMN evidence_metrics jsonb;
ALTER TABLE eval_runs ADD COLUMN slices jsonb;
ALTER TABLE eval_case_results ADD COLUMN evidence jsonb;
