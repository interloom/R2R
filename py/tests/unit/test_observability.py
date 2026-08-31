from core.utils.observability import (
    build_litellm_metadata,
    build_r2r_ingestion_trace_context,
    get_r2r_trace_context,
    r2r_observation_context,
    r2r_trace_context,
)


def test_builds_ingestion_trace_from_workflow_input():
    trace = build_r2r_ingestion_trace_context(
        {
            "document_id": "document-123",
            "langfuse_trace_id": "a" * 32,
        },
        task_id="task-456",
    )

    assert trace.trace_id == "a" * 32
    assert trace.trace_name == "R2R: Document ingestion"
    assert trace.tags == ("r2r", "r2r-ingestion")
    assert trace.metadata == {
        "document_id": "document-123",
        "r2r_task_id": "task-456",
    }


def test_merges_trace_and_observation_context_into_litellm_metadata():
    with r2r_trace_context(
        "R2R: Document ingestion",
        trace_id="b" * 32,
        tags=("r2r-ingestion",),
        metadata={"document_id": "document-123"},
    ):
        with r2r_observation_context(
            "R2R: Chunk enrichment",
            metadata={"r2r_chunk_id": "chunk-789"},
        ):
            metadata = build_litellm_metadata(
                {
                    "tags": ["existing-tag"],
                    "trace_metadata": {"request_source": "api"},
                    "custom": "value",
                },
                default_trace_name="R2R: LLM completion",
                default_generation_name="R2R: LLM completion",
            )

    assert metadata == {
        "tags": ["existing-tag", "r2r", "r2r-ingestion"],
        "trace_name": "R2R: Document ingestion",
        "generation_name": "R2R: Chunk enrichment",
        "trace_id": "b" * 32,
        "trace_metadata": {
            "document_id": "document-123",
            "request_source": "api",
        },
        "r2r_chunk_id": "chunk-789",
        "custom": "value",
    }


def test_preserves_explicit_litellm_names_and_trace_id():
    with r2r_trace_context(
        "R2R: Search",
        trace_id="c" * 32,
        tags=("r2r-search",),
    ):
        metadata = build_litellm_metadata(
            {
                "trace_id": "d" * 32,
                "trace_name": "Caller trace",
                "generation_name": "Caller generation",
            },
            default_trace_name="R2R: LLM completion",
            default_generation_name="R2R: LLM completion",
        )

    assert metadata["trace_id"] == "d" * 32
    assert metadata["trace_name"] == "Caller trace"
    assert metadata["generation_name"] == "Caller generation"
    assert metadata["tags"] == ["r2r", "r2r-search"]


def test_nested_trace_context_keeps_workflow_trace_and_restores_it():
    assert get_r2r_trace_context() is None

    with r2r_trace_context(
        "R2R: Document ingestion",
        trace_id="e" * 32,
        tags=("r2r-ingestion",),
        metadata={"document_id": "document-123"},
    ):
        with r2r_trace_context(
            "R2R: Search",
            tags=("r2r-search",),
            metadata={"search_strategy": "basic"},
        ) as nested:
            assert nested.trace_id == "e" * 32
            assert nested.trace_name == "R2R: Document ingestion"
            assert nested.tags == (
                "r2r",
                "r2r-ingestion",
                "r2r-search",
            )
            assert nested.metadata == {
                "document_id": "document-123",
                "search_strategy": "basic",
            }

        assert get_r2r_trace_context().trace_name == (
            "R2R: Document ingestion"
        )

    assert get_r2r_trace_context() is None
