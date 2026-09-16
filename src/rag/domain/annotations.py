from typing import Any, Protocol


class AnnotationConflictError(ValueError):
    """A note changed since the client's revision was read."""


class AnnotationStorePort(Protocol):
    def list_annotations(self, run_id: str) -> list[dict[str, Any]]: ...

    def save_annotation(
        self,
        run_id: str,
        case_id: str | None,
        note: str,
        revision: int,
    ) -> dict[str, Any]: ...
