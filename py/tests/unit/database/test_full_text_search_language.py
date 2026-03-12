from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from core.base import DatabaseConfig, SearchSettings, VectorQuantizationType
from core.providers.database.base import PostgresConnectionManager
from core.providers.database.chunks import PostgresChunksHandler
from core.providers.database.documents import PostgresDocumentsHandler
from core.providers.database.utils import psql_regconfig_literal


def test_database_config_defaults_full_text_language_to_english():
    assert DatabaseConfig().full_text_search_language == "english"


def test_psql_regconfig_literal_rejects_invalid_names():
    assert psql_regconfig_literal("simple") == "'simple'::regconfig"

    with pytest.raises(ValueError):
        psql_regconfig_literal("english'; DROP TABLE chunks; --")


@pytest.mark.asyncio
async def test_chunks_handler_uses_configured_full_text_language():
    fetch_query = AsyncMock(side_effect=[[], [], []])
    execute_query = AsyncMock()
    connection_manager = cast(
        PostgresConnectionManager,
        SimpleNamespace(fetch_query=fetch_query, execute_query=execute_query),
    )
    handler = PostgresChunksHandler(
        project_name="test_project",
        connection_manager=connection_manager,
        dimension=4,
        quantization_type=VectorQuantizationType.FP32,
        full_text_search_language="simple",
    )

    await handler.create_tables()
    await handler.full_text_search("foo", SearchSettings())

    assert execute_query.await_args is not None
    create_query = execute_query.await_args.args[0]
    assert "to_tsvector('simple'::regconfig, text)" in create_query

    assert fetch_query.await_args is not None
    search_query = fetch_query.await_args.args[0]
    assert "websearch_to_tsquery('simple'::regconfig, $1)" in search_query


@pytest.mark.asyncio
async def test_documents_handler_uses_configured_full_text_language():
    fetch_query = AsyncMock(
        side_effect=[[{"column_name": "total_tokens"}], []]
    )
    execute_query = AsyncMock()
    connection_manager = cast(
        PostgresConnectionManager,
        SimpleNamespace(fetch_query=fetch_query, execute_query=execute_query),
    )
    handler = PostgresDocumentsHandler(
        project_name="test_project",
        connection_manager=connection_manager,
        dimension=4,
        full_text_search_language="simple",
    )

    await handler.create_tables()
    await handler.full_text_document_search("foo", SearchSettings())

    assert execute_query.await_args is not None
    create_query = execute_query.await_args.args[0]
    assert "to_tsvector('simple'::regconfig, COALESCE(title, ''))" in create_query

    assert fetch_query.await_args is not None
    search_query = fetch_query.await_args.args[0]
    assert "websearch_to_tsquery('simple'::regconfig, $1)" in search_query
