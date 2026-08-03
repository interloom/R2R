from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from core.base import Vector, VectorEntry
from core.base.api.models import User
from core.main.services.ingestion_service import IngestionService


def _make_vector_entry(owner_id):
    return VectorEntry(
        id=uuid4(),
        document_id=uuid4(),
        owner_id=owner_id,
        collection_ids=[],
        vector=Vector(data=[0.1]),
        text="test chunk",
        metadata={},
    )


def _make_service(owner, current_usage=0, max_chunks=10_000):
    chunks_handler = SimpleNamespace(
        list_chunks=AsyncMock(
            return_value={"results": [], "total_entries": current_usage}
        ),
        upsert_entries=AsyncMock(),
    )
    users_handler = SimpleNamespace(
        get_user_by_id=AsyncMock(return_value=owner)
    )
    database = SimpleNamespace(
        chunks_handler=chunks_handler,
        users_handler=users_handler,
        config=SimpleNamespace(
            app=SimpleNamespace(
                default_max_chunks_per_user=max_chunks,
            )
        ),
    )
    providers = SimpleNamespace(database=database)
    service = IngestionService(config=Mock(), providers=providers)
    return service, chunks_handler, users_handler


@pytest.mark.asyncio
async def test_store_embeddings_skips_chunk_quota_for_superuser():
    owner = User(
        id=uuid4(),
        email="admin@example.com",
        is_superuser=True,
    )
    vector_entry = _make_vector_entry(owner.id)
    service, chunks_handler, users_handler = _make_service(
        owner,
        current_usage=10_000,
        max_chunks=1,
    )

    messages = [
        message async for message in service.store_embeddings([vector_entry])
    ]

    users_handler.get_user_by_id.assert_awaited_once_with(owner.id)
    chunks_handler.list_chunks.assert_not_awaited()
    chunks_handler.upsert_entries.assert_awaited_once_with([vector_entry])
    assert messages == [
        f"Successful ingestion for document_id: {vector_entry.document_id}, "
        "with vector count: 1"
    ]


@pytest.mark.asyncio
async def test_store_embeddings_keeps_chunk_quota_for_normal_user():
    owner = User(
        id=uuid4(),
        email="user@example.com",
        is_superuser=False,
    )
    vector_entry = _make_vector_entry(owner.id)
    service, chunks_handler, users_handler = _make_service(owner)

    messages = [
        message async for message in service.store_embeddings([vector_entry])
    ]

    users_handler.get_user_by_id.assert_awaited_once_with(owner.id)
    chunks_handler.list_chunks.assert_awaited_once_with(
        limit=1,
        offset=0,
        filters={"owner_id": owner.id},
    )
    chunks_handler.upsert_entries.assert_awaited_once_with([vector_entry])
    assert messages == [
        f"Successful ingestion for document_id: {vector_entry.document_id}, "
        "with vector count: 1"
    ]
