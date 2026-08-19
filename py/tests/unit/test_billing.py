from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from core.base import CompletionConfig, EmbeddingConfig, GenerationConfig
from core.billing import (
    BillingOutboxDispatcher,
    BillingUsageRecorder,
    InterloomBillingContext,
    with_billing_context_from_input,
)
from core.providers.embeddings.litellm import LiteLLMEmbeddingProvider
from core.providers.llm.litellm import LiteLLMCompletionProvider


def test_billing_context_requires_exactly_one_source():
    values = {
        "ingestion_id": uuid4(),
        "organization_id": uuid4(),
    }

    with pytest.raises(ValidationError):
        InterloomBillingContext(**values)

    with pytest.raises(ValidationError):
        InterloomBillingContext(
            **values,
            file_id=uuid4(),
            note_id=uuid4(),
        )


async def test_recorder_persists_usage_and_adds_reconciliation_metadata():
    handler = SimpleNamespace(
        create_pending=AsyncMock(),
        complete=AsyncMock(),
        fail=AsyncMock(),
    )
    recorder = BillingUsageRecorder(handler)
    context = InterloomBillingContext(
        ingestion_id=uuid4(),
        organization_id=uuid4(),
        file_id=uuid4(),
    )

    @with_billing_context_from_input
    async def record(input_data):
        event_id = await recorder.start_call(
            operation="completion",
            model="litellm_proxy/openai/gpt-5.4-nano",
        )
        kwargs = recorder.add_litellm_metadata(
            kwargs={"metadata": {"existing": "value"}},
            event_id=event_id,
        )
        await recorder.complete_call(
            event_id,
            {
                "id": "provider-request-id",
                "model": "openai/gpt-5.4-nano",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 3},
                },
            },
        )
        return event_id, kwargs

    event_id, kwargs = await record(
        {"interloom_billing_context": context.model_dump(mode="json")}
    )

    handler.create_pending.assert_awaited_once()
    handler.complete.assert_awaited_once_with(
        event_id=event_id,
        provider_request_id="provider-request-id",
        resolved_model="openai/gpt-5.4-nano",
        ended_at=handler.complete.await_args.kwargs["ended_at"],
        input_tokens=10,
        output_tokens=4,
        cached_input_tokens=3,
    )
    assert kwargs["metadata"]["existing"] == "value"
    assert kwargs["metadata"]["spend_logs_metadata"] == {
        "source": "r2r",
        "billing_event_id": str(event_id),
        "ingestion_id": str(context.ingestion_id),
        "organization_id": str(context.organization_id),
        "file_id": str(context.file_id),
        "note_id": None,
    }

    assert (
        await recorder.start_call(operation="embedding", model="model") is None
    )


async def test_dispatcher_marks_successful_delivery():
    event_id = uuid4()
    handler = SimpleNamespace(
        claim_for_delivery=AsyncMock(
            return_value=[
                {"billing_event_id": str(event_id), "input_tokens": 10}
            ]
        ),
        mark_delivered=AsyncMock(),
        mark_delivery_failed=AsyncMock(),
    )
    dispatcher = BillingOutboxDispatcher(
        outbox_handler=handler,
        callback_url="https://interloom.example.test/r2r-usage",
        service_token="service-token",
    )

    async def send(request: httpx.Request) -> httpx.Response:
        assert request.headers["il-service-token"] == "service-token"
        return httpx.Response(200)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(send)
    ) as client:
        await dispatcher._deliver_batch(client)

    handler.mark_delivered.assert_awaited_once_with(event_id=str(event_id))
    handler.mark_delivery_failed.assert_not_awaited()


async def test_litellm_providers_record_completion_and_embedding_usage():
    event_id = uuid4()
    recorder = SimpleNamespace(
        start_call=AsyncMock(return_value=event_id),
        add_litellm_metadata=Mock(
            side_effect=lambda *, kwargs, event_id: {
                **kwargs,
                "metadata": {"billing_event_id": str(event_id)},
            }
        ),
        complete_call=AsyncMock(),
        fail_call=AsyncMock(),
    )
    response = SimpleNamespace(
        id="provider-request-id",
        model="openai/model",
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        data=[{"embedding": [0.1, 0.2]}],
    )

    completion_provider = LiteLLMCompletionProvider(
        CompletionConfig(provider="litellm"),
        billing_usage_recorder=recorder,
    )
    completion_provider.acompletion = AsyncMock(return_value=response)
    completion_result = await completion_provider._execute_task(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "generation_config": GenerationConfig(model="openai/model"),
            "kwargs": {},
        }
    )

    assert completion_result is response
    assert completion_provider.acompletion.await_args.kwargs["metadata"] == {
        "billing_event_id": str(event_id)
    }
    recorder.complete_call.assert_awaited_with(event_id, response)

    embedding_provider = LiteLLMEmbeddingProvider(
        EmbeddingConfig(
            provider="litellm",
            base_model="openai/embedding-model",
            base_dimension=2,
        ),
        billing_usage_recorder=recorder,
    )
    embedding_provider.litellm_aembedding = AsyncMock(return_value=response)
    embedding_result = await embedding_provider._execute_task(
        {"texts": ["hello"], "kwargs": {}}
    )

    assert embedding_result == [[0.1, 0.2]]
    assert embedding_provider.litellm_aembedding.await_args.kwargs[
        "metadata"
    ] == {"billing_event_id": str(event_id)}
    recorder.complete_call.assert_awaited_with(event_id, response)
