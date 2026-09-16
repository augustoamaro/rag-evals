from __future__ import annotations

from typing import Any

from rag.adapters.db.pool import Pool
from rag.domain.annotations import AnnotationConflictError


class AnnotationStore:
    """Review notes stay separate from immutable benchmark inputs and observations."""

    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    def list_annotations(self, run_id: str) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            if (
                conn.execute(
                    "SELECT 1 FROM eval_runs WHERE id = %s", (run_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(run_id)
            rows = conn.execute(
                "SELECT case_id, note, revision, updated_at FROM eval_annotations "
                "WHERE run_id = %s ORDER BY case_id",
                (run_id,),
            ).fetchall()
        return [_note(row) for row in rows]

    def save_annotation(
        self,
        run_id: str,
        case_id: str | None,
        note: str,
        revision: int,
    ) -> dict[str, Any]:
        if len(note) > 4000 or revision < 0:
            raise ValueError("invalid annotation length or revision")
        with self._pool.connection() as conn:
            # Retain the parent while checking case ownership and writing a note.
            parent = conn.execute(
                "SELECT id FROM eval_runs WHERE id = %s FOR KEY SHARE",
                (run_id,),
            ).fetchone()
            if parent is None:
                raise KeyError(run_id)
            if (
                case_id is not None
                and conn.execute(
                    "SELECT 1 FROM eval_case_results WHERE run_id = %s AND case_id = %s",
                    (run_id, case_id),
                ).fetchone()
                is None
            ):
                raise KeyError(case_id)
            if revision == 0:
                row = conn.execute(
                    "INSERT INTO eval_annotations (run_id, case_id, note, revision) "
                    "VALUES (%s, %s, %s, 1) ON CONFLICT (run_id, case_id) DO NOTHING "
                    "RETURNING case_id, note, revision, updated_at",
                    (run_id, case_id or "", note),
                ).fetchone()
            else:
                row = conn.execute(
                    "UPDATE eval_annotations SET note = %s, revision = revision + 1, "
                    "updated_at = clock_timestamp() WHERE run_id = %s AND case_id = %s "
                    "AND revision = %s RETURNING case_id, note, revision, updated_at",
                    (note, run_id, case_id or "", revision),
                ).fetchone()
            if row is None:
                raise AnnotationConflictError("annotation revision changed")
        return _note(row)


def _note(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "case_id": row[0] or None,
        "note": row[1],
        "revision": row[2],
        "updated_at": row[3].isoformat(),
    }
