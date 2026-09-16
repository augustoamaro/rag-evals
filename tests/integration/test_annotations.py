from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

from rag.adapters.db.annotation_store import AnnotationStore
from rag.adapters.db.eval_store import EvalStore
from rag.adapters.db.pool import Pool
from rag.api.app import AppDeps, create_app
from rag.domain.annotations import AnnotationConflictError
from rag.domain.entities import Strategy
from rag.services.eval_service import EvalService
from tests.unit.test_api import FakeEvalService, FakeRetriever
from tests.unit.test_versioned_eval import fixture

pytestmark = pytest.mark.integration


def seed(pool: Pool) -> str:
    dataset, index = fixture()
    run = EvalService(index, index, dataset=dataset).run(
        dataset.cases, Strategy.HYBRID, 2
    )
    EvalStore(pool).save_run(run, run.manifest)
    return run.id


def test_notes_are_independent_of_results_and_preserve_revisions(pool: Pool) -> None:
    run_id = seed(pool)
    store = AnnotationStore(pool)
    before = EvalStore(pool).get_run(run_id)
    assert store.list_annotations(run_id) == []
    first = store.save_annotation(run_id, None, "Inspect failure <script>", 0)
    assert first["revision"] == 1 and first["case_id"] is None
    store.save_annotation(run_id, "q", "Case observation", 0)
    cleared = store.save_annotation(run_id, None, "", 1)
    assert cleared["revision"] == 2 and cleared["note"] == ""
    with pytest.raises(AnnotationConflictError):
        store.save_annotation(run_id, None, "stale overwrite", 1)
    assert len(store.list_annotations(run_id)) == 2
    assert EvalStore(pool).get_run(run_id) == before


def test_unknown_run_or_case_cannot_receive_notes(pool: Pool) -> None:
    run_id = seed(pool)
    store = AnnotationStore(pool)
    with pytest.raises(KeyError):
        store.list_annotations("absent")
    with pytest.raises(KeyError):
        store.save_annotation("absent", None, "note", 0)
    with pytest.raises(KeyError):
        store.save_annotation(run_id, "absent", "note", 0)
    assert not store.list_annotations(run_id)


@pytest.mark.parametrize("initial", [False, True])
def test_concurrent_note_edits_never_silently_overwrite(
    pool: Pool, initial: bool
) -> None:
    run_id = seed(pool)
    store = AnnotationStore(pool)
    if initial:
        store.save_annotation(run_id, None, "original", 0)
    barrier = Barrier(2)

    def write(note: str) -> str:
        barrier.wait(timeout=5)
        try:
            store.save_annotation(run_id, None, note, int(initial))
        except AnnotationConflictError:
            return "conflict"
        return note

    with ThreadPoolExecutor(2) as workers:
        results = list(workers.map(write, ["first", "second"]))
    assert results.count("conflict") == 1
    notes = store.list_annotations(run_id)
    assert notes[0]["note"] in results
    assert notes[0]["revision"] == int(initial) + 1


def test_annotation_api_validation_conflict_and_missing_case(pool: Pool) -> None:
    run_id = seed(pool)
    api = TestClient(
        create_app(
            AppDeps(
                FakeRetriever(),
                None,
                FakeEvalService(),
                EvalStore(pool),
                [],
                annotations=AnnotationStore(pool),
            )
        )
    )
    endpoint = f"/evals/runs/{run_id}/annotations"
    assert api.get(endpoint).json() == []
    response = api.put(endpoint, json={"note": "reviewed", "revision": 0})
    assert response.status_code == 200 and response.json()["revision"] == 1
    assert api.put(endpoint, json={"note": "conflict", "revision": 0}).status_code == 409
    assert api.put(endpoint, json={"note": "x" * 4001, "revision": 1}).status_code == 422
    assert api.put(endpoint, json={"note": "x", "revision": -1}).status_code == 422
    assert (
        api.put(
            endpoint, json={"case_id": "no-case", "note": "x", "revision": 0}
        ).status_code
        == 404
    )
    assert api.get("/evals/runs/absent/annotations").status_code == 404
