-- Historical completeness is unknown; all new evaluations record it explicitly.
ALTER TABLE eval_runs ADD COLUMN retrieval_complete boolean;
ALTER TABLE eval_runs ADD COLUMN error_type text;

CREATE UNIQUE INDEX eval_case_results_run_case
  ON eval_case_results (run_id, case_id);

CREATE TABLE eval_jobs (
  id text PRIMARY KEY,
  run_id text UNIQUE REFERENCES eval_runs (id) ON DELETE CASCADE,
  request_id uuid NOT NULL UNIQUE,
  config jsonb NOT NULL,
  status text NOT NULL CHECK (status IN (
    'queued', 'running', 'cancelling', 'completed', 'partial', 'failed', 'cancelled'
  )),
  total integer NOT NULL CHECK (total > 0),
  completed integer NOT NULL DEFAULT 0 CHECK (completed >= 0 AND completed <= total),
  stage text NOT NULL DEFAULT 'queued',
  case_id text,
  error_type text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- Admission is bounded even when competing HTTP processes race.
CREATE UNIQUE INDEX eval_jobs_one_active ON eval_jobs ((true))
  WHERE status IN ('queued', 'running', 'cancelling');
CREATE INDEX eval_jobs_recent ON eval_jobs (created_at DESC);

CREATE TABLE eval_annotations (
  run_id text NOT NULL REFERENCES eval_runs (id) ON DELETE CASCADE,
  case_id text NOT NULL DEFAULT '',
  note text NOT NULL CHECK (length(note) <= 4000),
  revision integer NOT NULL DEFAULT 1 CHECK (revision > 0),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, case_id)
);
