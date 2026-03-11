import os
import zipfile
from io import BytesIO
from uuid import uuid4

import pytest

from core.base import R2RException
from core.base.providers.file import FileConfig
from core.providers.file import AzureBlobFileProvider


AZURITE_ENABLED = os.getenv("AZURE_BLOB_TEST_ENABLED") == "1"
AZURITE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURITE_CONTAINER_NAME = os.getenv(
    "AZURE_STORAGE_CONTAINER_NAME", "r2r-azure-test"
)


def _create_provider() -> AzureBlobFileProvider:
    if not AZURITE_ENABLED or not AZURITE_CONNECTION_STRING:
        pytest.skip("Azurite-backed Azure Blob tests are not enabled")

    config = FileConfig(
        provider="azure_blob",
        container_name=AZURITE_CONTAINER_NAME,
        connection_string=AZURITE_CONNECTION_STRING,
    )
    return AzureBlobFileProvider(config)


@pytest.mark.asyncio
async def test_azure_blob_roundtrip_and_overview():
    provider = _create_provider()
    await provider.initialize()

    document_id = uuid4()
    file_name = "azure-roundtrip.txt"
    file_content = b"hello from azurite"

    try:
        await provider.store_file(
            document_id=document_id,
            file_name=file_name,
            file_content=BytesIO(file_content),
            file_type="text/plain",
        )

        retrieved_name, retrieved_file, retrieved_size = await provider.retrieve_file(
            document_id
        ) or (None, None, None)

        assert retrieved_name == file_name
        assert retrieved_size == len(file_content)
        assert retrieved_file is not None
        assert retrieved_file.read() == file_content

        overview = await provider.get_files_overview(
            offset=0,
            limit=10,
            filter_document_ids=[document_id],
        )

        assert len(overview) == 1
        assert overview[0]["file_name"] == file_name
        assert overview[0]["document_id"] == document_id
    finally:
        await provider.delete_file(document_id)


@pytest.mark.asyncio
async def test_azure_blob_zip_export_and_delete():
    provider = _create_provider()
    await provider.initialize()

    document_id = uuid4()
    file_name = "azure-zip.txt"
    file_content = b"zip me"

    await provider.store_file(
        document_id=document_id,
        file_name=file_name,
        file_content=BytesIO(file_content),
        file_type="text/plain",
    )

    zip_name, zip_file, zip_size = await provider.retrieve_files_as_zip(
        [document_id]
    )

    assert zip_name.endswith(".zip")
    assert zip_size > 0

    with zipfile.ZipFile(zip_file) as archive:
        assert archive.namelist() == [file_name]
        assert archive.read(file_name) == file_content

    await provider.delete_file(document_id)

    with pytest.raises(R2RException) as exc_info:
        await provider.retrieve_file(document_id)

    assert exc_info.value.status_code == 404
