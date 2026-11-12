from rag.metrics.citations import cited_ids, split_citations


def test_cited_ids_are_unique_and_in_first_seen_order() -> None:
    text = "A [rag:0], then [cap:1] and [rag:0] again. Not an id: [two words]."
    assert cited_ids(text) == ["rag:0", "cap:1"]


def test_split_separates_retrieved_passages_from_unknown_ids() -> None:
    valid, invalid = split_citations(
        "Claim [cap:0][doc:0], more [rag:1] and [doc:cap:0].", {"cap:0", "rag:1", "x:9"}
    )
    assert valid == ["cap:0", "rag:1"]
    assert invalid == ["doc:0", "doc:cap:0"]


def test_text_without_citations_has_neither() -> None:
    assert split_citations("No brackets here.", {"cap:0"}) == ([], [])
