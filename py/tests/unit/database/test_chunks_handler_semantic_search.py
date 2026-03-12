from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from core.base import SearchSettings, VectorQuantizationType
from core.providers.database.chunks import PostgresChunksHandler
from core.providers.database.base import PostgresConnectionManager


@pytest.mark.asyncio
async def test_semantic_search_materializes_filtered_fp32_results():
    fetch_query = AsyncMock(return_value=[])
    connection_manager = cast(
        PostgresConnectionManager,
        SimpleNamespace(fetch_query=fetch_query),
    )
    handler = PostgresChunksHandler(
        project_name="test_project",
        connection_manager=connection_manager,
        dimension=4,
        quantization_type=VectorQuantizationType.FP32,
    )
    settings = SearchSettings(
        filters={"document_id": {"$in": ["11111111-1111-4111-8111-111111111111"]}},
        limit=10,
        offset=0,
    )

    await handler.semantic_search([0.1, 0.2, 0.3, 0.4], settings)

    assert fetch_query.await_args is not None
    query = fetch_query.await_args.args[0]
    assert "WITH filtered AS MATERIALIZED" in query
    assert "FROM filtered" in query
    assert "ORDER BY vec <=> $1::vector(4)" in query
    assert "test_project.chunks.id" not in query.split("FROM filtered", 1)[1]


@pytest.mark.asyncio
async def test_semantic_search_keeps_direct_knn_path_without_filters():
    fetch_query = AsyncMock(return_value=[])
    connection_manager = cast(
        PostgresConnectionManager,
        SimpleNamespace(fetch_query=fetch_query),
    )
    handler = PostgresChunksHandler(
        project_name="test_project",
        connection_manager=connection_manager,
        dimension=4,
        quantization_type=VectorQuantizationType.FP32,
    )
    settings = SearchSettings(limit=10, offset=0)

    await handler.semantic_search([0.1, 0.2, 0.3, 0.4], settings)

    assert fetch_query.await_args is not None
    query = fetch_query.await_args.args[0]
    assert "WITH filtered AS MATERIALIZED" not in query
    assert "FROM filtered" not in query
    assert 'FROM "test_project"."chunks"' in query


@pytest.mark.asyncio
async def test_semantic_search_materializes_filtered_int1_candidates():
    fetch_query = AsyncMock(return_value=[])
    connection_manager = cast(
        PostgresConnectionManager,
        SimpleNamespace(fetch_query=fetch_query),
    )
    handler = PostgresChunksHandler(
        project_name="test_project",
        connection_manager=connection_manager,
        dimension=4,
        quantization_type=VectorQuantizationType.INT1,
    )
    settings = SearchSettings(
        filters={"document_id": {"$in": ["11111111-1111-4111-8111-111111111111"]}},
        limit=10,
        offset=0,
    )

    await handler.semantic_search([0.1, 0.2, 0.3, 0.4], settings)

    assert fetch_query.await_args is not None
    query = fetch_query.await_args.args[0]
    assert "WITH filtered AS MATERIALIZED" in query
    assert "FROM filtered" in query
    assert "vec_binary" in query
    assert "ORDER BY distance" in query
