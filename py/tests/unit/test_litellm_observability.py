from unittest.mock import AsyncMock

import pytest
from core.base import EmbeddingConfig, GenerationConfig
from core.base.providers.llm import CompletionConfig
from core.providers.embeddings.litellm import LiteLLMEmbeddingProvider
from core.providers.llm.litellm import LiteLLMCompletionProvider
from core.utils.observability import (
    r2r_observation_context,
    r2r_trace_context,
)


@pytest.mark.asyncio
async def test_completion_provider_adds_current_langfuse_context():
    provider = LiteLLMCompletionProvider(CompletionConfig(provider="litellm"))
    provider.acompletion = AsyncMock(return_value="completion")

    with r2r_trace_context(
        "R2R: Document ingestion",
        trace_id="a" * 32,
        tags=("r2r-ingestion",),
        metadata={"document_id": "document-123"},
    ):
        with r2r_observation_context("R2R: Document summary"):
            result = await provider._execute_task(
                {
                    "messages": [{"role": "user", "content": "Summarize"}],
                    "generation_config": GenerationConfig(model="test-model"),
                    "kwargs": {},
                }
            )

    assert result == "completion"
    metadata = provider.acompletion.await_args.kwargs["metadata"]
    assert metadata == {
        "tags": ["r2r", "r2r-ingestion"],
        "trace_name": "R2R: Document ingestion",
        "generation_name": "R2R: Document summary",
        "trace_id": "a" * 32,
        "trace_metadata": {"document_id": "document-123"},
    }


def test_embedding_provider_adds_r2r_fallback_metadata():
    provider = LiteLLMEmbeddingProvider(
        EmbeddingConfig(
            provider="litellm",
            base_model="test-embedding-model",
            base_dimension=3,
        )
    )

    kwargs = provider._get_embedding_kwargs(
        metadata={"tags": ["existing-tag"]}
    )

    assert kwargs["metadata"] == {
        "tags": ["existing-tag", "r2r"],
        "trace_name": "R2R: Embedding",
        "generation_name": "R2R: Embedding",
    }
