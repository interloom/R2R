import asyncio
import contextvars
import functools
import logging
import os
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)


class InterloomBillingContext(BaseModel):
    ingestion_id: UUID
    organization_id: UUID
    file_id: UUID | None = None
    note_id: UUID | None = None

    @model_validator(mode="after")
    def validate_source(self):
        if (self.file_id is None) == (self.note_id is None):
            raise ValueError("Exactly one of file_id or note_id is required")
        return self

    @property
    def source_id(self) -> UUID:
        if self.file_id is not None:
            return self.file_id
        if self.note_id is None:
            raise ValueError("Billing context source is missing")
        return self.note_id


_billing_context: contextvars.ContextVar[InterloomBillingContext | None] = (
    contextvars.ContextVar("interloom_billing_context", default=None)
)


def _context_from_input(
    input_data: dict[str, Any],
) -> InterloomBillingContext | None:
    value = input_data.get("interloom_billing_context")
    if value is None:
        return None
    return InterloomBillingContext.model_validate(value)


def with_billing_context_from_input(function: Callable[..., Awaitable[Any]]):
    @functools.wraps(function)
    async def wrapped(input_data: dict[str, Any], *args: Any, **kwargs: Any):
        token = _billing_context.set(_context_from_input(input_data))
        try:
            return await function(input_data, *args, **kwargs)
        finally:
            _billing_context.reset(token)

    return wrapped


def with_billing_context_from_hatchet(function: Callable[..., Awaitable[Any]]):
    @functools.wraps(function)
    async def wrapped(instance: Any, context: Any, *args: Any, **kwargs: Any):
        input_data = context.workflow_input()["request"]
        token = _billing_context.set(_context_from_input(input_data))
        try:
            return await function(instance, context, *args, **kwargs)
        finally:
            _billing_context.reset(token)

    return wrapped


def _field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _usage_from_response(response: Any) -> dict[str, int | None]:
    usage = _field(response, "usage")
    input_tokens = _field(usage, "prompt_tokens")
    if input_tokens is None:
        input_tokens = _field(usage, "input_tokens")
    output_tokens = _field(usage, "completion_tokens")
    if output_tokens is None:
        output_tokens = _field(usage, "output_tokens")

    input_details = _field(usage, "prompt_tokens_details")
    if input_details is None:
        input_details = _field(usage, "input_tokens_details")

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": _field(input_details, "cached_tokens"),
    }


class BillingUsageRecorder:
    def __init__(self, outbox_handler: Any):
        self.outbox_handler = outbox_handler

    async def start_call(self, *, operation: str, model: str) -> UUID | None:
        context = _billing_context.get()
        if context is None:
            return None

        event_id = uuid4()
        await self.outbox_handler.create_pending(
            event_id=event_id,
            context=context,
            operation=operation,
            requested_model=model,
            started_at=datetime.now(timezone.utc),
        )
        return event_id

    def add_litellm_metadata(
        self,
        *,
        kwargs: dict[str, Any],
        event_id: UUID | None,
    ) -> dict[str, Any]:
        if event_id is None:
            return kwargs

        context = _billing_context.get()
        if context is None:
            return kwargs

        result = dict(kwargs)
        metadata = dict(result.get("metadata") or {})
        spend_logs_metadata = dict(metadata.get("spend_logs_metadata") or {})
        spend_logs_metadata.update(
            {
                "source": "r2r",
                "billing_event_id": str(event_id),
                "ingestion_id": str(context.ingestion_id),
                "organization_id": str(context.organization_id),
                "file_id": str(context.file_id) if context.file_id else None,
                "note_id": str(context.note_id) if context.note_id else None,
            }
        )
        metadata["spend_logs_metadata"] = spend_logs_metadata
        result["metadata"] = metadata
        return result

    async def complete_call(
        self, event_id: UUID | None, response: Any
    ) -> None:
        if event_id is None:
            return
        await self.outbox_handler.complete(
            event_id=event_id,
            provider_request_id=_field(response, "id"),
            resolved_model=_field(response, "model"),
            ended_at=datetime.now(timezone.utc),
            **_usage_from_response(response),
        )

    async def fail_call(self, event_id: UUID | None, error: Exception) -> None:
        if event_id is None:
            return
        await self.outbox_handler.fail(
            event_id=event_id,
            ended_at=datetime.now(timezone.utc),
            error=str(error),
        )


class BillingOutboxDispatcher:
    def __init__(
        self,
        *,
        outbox_handler: Any,
        callback_url: str,
        service_token: str,
        poll_interval_seconds: float = 5,
    ):
        self.outbox_handler = outbox_handler
        self.callback_url = callback_url
        self.service_token = service_token
        self.poll_interval_seconds = poll_interval_seconds
        self._stopped = asyncio.Event()

    @classmethod
    def from_environment(cls, outbox_handler: Any):
        callback_url = os.getenv("R2R_BILLING_CALLBACK_URL")
        service_token = os.getenv("R2R_BILLING_CALLBACK_SERVICE_TOKEN")
        if not callback_url or not service_token:
            return None
        return cls(
            outbox_handler=outbox_handler,
            callback_url=callback_url,
            service_token=service_token,
        )

    async def run(self) -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            while not self._stopped.is_set():
                try:
                    await self._deliver_batch(client)
                except Exception:
                    logger.exception("Failed to dispatch R2R billing usage")

                try:
                    await asyncio.wait_for(
                        self._stopped.wait(),
                        timeout=self.poll_interval_seconds,
                    )
                except TimeoutError:
                    pass

    async def stop(self) -> None:
        self._stopped.set()

    async def _deliver_batch(self, client: httpx.AsyncClient) -> None:
        events = await self.outbox_handler.claim_for_delivery(limit=100)
        for event in events:
            event_id = event["billing_event_id"]
            try:
                response = await client.post(
                    self.callback_url,
                    json=event,
                    headers={"il-service-token": self.service_token},
                )
                response.raise_for_status()
            except Exception as error:
                await self.outbox_handler.mark_delivery_failed(
                    event_id=event_id,
                    error=str(error),
                )
            else:
                await self.outbox_handler.mark_delivered(event_id=event_id)
