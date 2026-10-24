"""Fakes for the deep research retrieval seam (no worker process, corpus, or model)."""


def patch_retrieval(monkeypatch, deep_module, retrieve):
    """Route both the per-viewpoint and the batched round retrieval through `retrieve`.

    `retrieve(query, chunks, **kwargs)` returns one schema-2 response. The
    batched seam calls it once per query in order, so tests written against
    single-viewpoint retrieval keep observing every query.
    """
    monkeypatch.setattr(deep_module, "_retrieve_candidates", retrieve)
    monkeypatch.setattr(
        deep_module,
        "_retrieve_candidates_many",
        lambda queries, chunks, **kwargs: [retrieve(query, chunks, **kwargs) for query in queries],
    )
