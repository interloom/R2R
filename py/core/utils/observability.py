from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping
from uuid import uuid4

R2R_TAG = "r2r"


@dataclass(frozen=True)
class R2RTraceContext:
    trace_id: str
    trace_name: str
    tags: tuple[str, ...] = (R2R_TAG,)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    session_id: str | None = None


@dataclass(frozen=True)
class R2RObservationContext:
    generation_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


_trace_context: ContextVar[R2RTraceContext | None] = ContextVar(
    "r2r_trace_context", default=None
)
_observation_context: ContextVar[R2RObservationContext | None] = ContextVar(
    "r2r_observation_context", default=None
)


def create_r2r_trace_id() -> str:
    """Create a Langfuse-compatible 32-character hexadecimal trace ID."""
    return uuid4().hex


def get_r2r_trace_context() -> R2RTraceContext | None:
    return _trace_context.get()


def get_r2r_observation_context() -> R2RObservationContext | None:
    return _observation_context.get()


def set_r2r_trace_context(context: R2RTraceContext) -> Token:
    return _trace_context.set(context)


def reset_r2r_trace_context(token: Token) -> None:
    _trace_context.reset(token)


def _unique_tags(*tag_groups: object) -> list[str]:
    tags: list[str] = []
    for group in tag_groups:
        values = [group] if isinstance(group, str) else group
        if not isinstance(values, (list, tuple, set, frozenset)):
            continue
        for value in values:
            if isinstance(value, str) and value not in tags:
                tags.append(value)
    return tags


@contextmanager
def r2r_trace_context(
    trace_name: str,
    *,
    trace_id: str | None = None,
    tags: tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
    session_id: str | None = None,
) -> Iterator[R2RTraceContext]:
    """Set trace-level attributes inherited by nested R2R model calls."""
    current = get_r2r_trace_context()
    context = R2RTraceContext(
        trace_id=(current.trace_id if current else trace_id)
        or create_r2r_trace_id(),
        trace_name=current.trace_name if current else trace_name,
        tags=tuple(
            _unique_tags(
                current.tags if current else (),
                (R2R_TAG,),
                tags,
            )
        ),
        metadata={
            **(dict(current.metadata) if current else {}),
            **dict(metadata or {}),
        },
        session_id=(current.session_id if current else None) or session_id,
    )
    token = _trace_context.set(context)
    try:
        yield context
    finally:
        _trace_context.reset(token)


@contextmanager
def r2r_observation_context(
    generation_name: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[R2RObservationContext]:
    """Set operation-level attributes for one nested R2R model call."""
    context = R2RObservationContext(
        generation_name=generation_name,
        metadata=dict(metadata or {}),
    )
    token = _observation_context.set(context)
    try:
        yield context
    finally:
        _observation_context.reset(token)


@contextmanager
def r2r_ingestion_trace_context(
    input_data: Mapping[str, Any],
    *,
    task_id: str | None = None,
) -> Iterator[R2RTraceContext]:
    """Create the shared trace used by one document-ingestion attempt."""
    context = build_r2r_ingestion_trace_context(
        input_data,
        task_id=task_id,
    )
    token = set_r2r_trace_context(context)
    try:
        yield context
    finally:
        reset_r2r_trace_context(token)


def build_r2r_ingestion_trace_context(
    input_data: Mapping[str, Any],
    *,
    task_id: str | None = None,
) -> R2RTraceContext:
    """Build the trace attributes for one document-ingestion attempt."""
    document_id = str(input_data["document_id"])
    trace_metadata = {"document_id": document_id}
    if task_id:
        trace_metadata["r2r_task_id"] = task_id
    supplied_trace_id = input_data.get("langfuse_trace_id")
    return R2RTraceContext(
        trace_id=(
            str(supplied_trace_id)
            if supplied_trace_id
            else create_r2r_trace_id()
        ),
        trace_name="R2R: Document ingestion",
        tags=(R2R_TAG, "r2r-ingestion"),
        metadata=trace_metadata,
    )


def build_litellm_metadata(
    existing: Mapping[str, Any] | None = None,
    *,
    default_trace_name: str,
    default_generation_name: str,
) -> dict[str, Any]:
    """Merge R2R trace context into LiteLLM's Langfuse metadata fields."""
    metadata = dict(existing or {})
    trace = get_r2r_trace_context()
    observation = get_r2r_observation_context()

    metadata["tags"] = _unique_tags(
        metadata.get("tags", ()),
        trace.tags if trace else (),
        (R2R_TAG,),
    )
    metadata.setdefault(
        "trace_name", trace.trace_name if trace else default_trace_name
    )
    metadata.setdefault(
        "generation_name",
        (
            observation.generation_name
            if observation
            else default_generation_name
        ),
    )

    if trace:
        metadata.setdefault("trace_id", trace.trace_id)
        if trace.session_id:
            metadata.setdefault("session_id", trace.session_id)

        trace_metadata = dict(trace.metadata)
        existing_trace_metadata = metadata.get("trace_metadata")
        if isinstance(existing_trace_metadata, Mapping):
            trace_metadata.update(existing_trace_metadata)
        if trace_metadata:
            metadata["trace_metadata"] = trace_metadata

    if observation:
        for key, value in observation.metadata.items():
            metadata.setdefault(key, value)

    return metadata
