import json
import zipfile
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest

from core.base import R2RException
from core.base.providers.file import FileConfig
from core.main.assembly.factory import R2RProviderFactory
from core.providers.file import LocalFileProvider


def _metadata_path(base_path: Path, document_id) -> Path:
    return base_path / "documents" / str(document_id) / "metadata.json"


def _update_created_at(
    base_path: Path, document_id, created_at: datetime
) -> None:
    metadata_path = _metadata_path(base_path, document_id)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    iso_timestamp = created_at.isoformat()
    metadata["created_at"] = iso_timestamp
    metadata["updated_at"] = iso_timestamp
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_local_file_config_supports_provider(tmp_path: Path):
    config = FileConfig(provider="local", base_path=str(tmp_path))

    config.validate_config()

    assert "local" in config.supported_providers


def test_local_file_config_requires_base_path(monkeypatch):
    monkeypatch.delenv("LOCAL_FILE_STORAGE_PATH", raising=False)
    config = FileConfig(provider="local")

    with pytest.raises(ValueError, match="base path"):
        config.validate_config()


def test_factory_creates_local_provider(tmp_path: Path):
    config = FileConfig(provider="local", base_path=str(tmp_path))

    provider = R2RProviderFactory.create_file_provider(config)

    assert isinstance(provider, LocalFileProvider)


@pytest.mark.asyncio
async def test_local_file_provider_roundtrip_and_overview(tmp_path: Path):
    provider = LocalFileProvider(
        FileConfig(provider="local", base_path=str(tmp_path))
    )
    await provider.initialize()

    document_id = uuid4()
    payload = b"hello local storage"
    await provider.store_file(
        document_id=document_id,
        file_name="hello.txt",
        file_content=BytesIO(payload),
        file_type="text/plain",
    )

    file_name, file_content, file_size = await provider.retrieve_file(
        document_id
    ) or (
        None,
        None,
        None,
    )

    assert file_name == "hello.txt"
    assert file_size == len(payload)
    assert file_content is not None
    assert file_content.read() == payload

    overview = await provider.get_files_overview(
        offset=0,
        limit=10,
        filter_document_ids=[document_id],
    )
    assert len(overview) == 1
    assert overview[0]["document_id"] == document_id
    assert overview[0]["file_name"] == "hello.txt"
    assert overview[0]["file_key"] == f"documents/{document_id}/content"
    assert overview[0]["file_type"] == "text/plain"


@pytest.mark.asyncio
async def test_local_file_provider_zip_export_filters_by_date(tmp_path: Path):
    provider = LocalFileProvider(
        FileConfig(provider="local", base_path=str(tmp_path))
    )
    await provider.initialize()

    older_document_id = uuid4()
    newer_document_id = uuid4()

    await provider.store_file(
        document_id=older_document_id,
        file_name="older.txt",
        file_content=BytesIO(b"old"),
        file_type="text/plain",
    )
    await provider.store_file(
        document_id=newer_document_id,
        file_name="newer.txt",
        file_content=BytesIO(b"new"),
        file_type="text/plain",
    )

    now = datetime.now(timezone.utc)
    _update_created_at(tmp_path, older_document_id, now - timedelta(days=2))
    _update_created_at(tmp_path, newer_document_id, now)

    _, zip_content, _ = await provider.retrieve_files_as_zip(
        start_date=now - timedelta(hours=12)
    )

    with zipfile.ZipFile(zip_content) as archive:
        assert archive.namelist() == ["newer.txt"]
        assert archive.read("newer.txt") == b"new"


@pytest.mark.asyncio
async def test_local_file_provider_delete_and_missing_file(tmp_path: Path):
    provider = LocalFileProvider(
        FileConfig(provider="local", base_path=str(tmp_path))
    )
    await provider.initialize()

    document_id = uuid4()
    await provider.store_file(
        document_id=document_id,
        file_name="delete-me.txt",
        file_content=BytesIO(b"delete me"),
        file_type="text/plain",
    )

    assert await provider.delete_file(document_id) is True

    with pytest.raises(R2RException, match="not found") as exc_info:
        await provider.retrieve_file(document_id)
    assert exc_info.value.status_code == 404

    with pytest.raises(R2RException, match="No files found"):
        await provider.get_files_overview(offset=0, limit=10)
