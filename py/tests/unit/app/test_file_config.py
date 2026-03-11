import asyncio

import pytest

from core.base.providers.file import FileConfig
from core.base import R2RException
from core.main.assembly.factory import R2RProviderFactory


def test_file_config_supports_azure_blob():
    config = FileConfig(
        provider="azure_blob",
        container_name="documents",
        connection_string="UseDevelopmentStorage=true",
    )

    config.validate_config()
    assert "azure_blob" in config.supported_providers


def test_file_config_requires_azure_connection_string(monkeypatch):
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    config = FileConfig(provider="azure_blob", container_name="documents")

    with pytest.raises(ValueError, match="connection string"):
        config.validate_config()


def test_factory_creates_azure_blob_provider(monkeypatch):
    class FakeBlobServiceClient:
        connection_string: str

        def __init__(self):
            self.connection_string = ""

        @classmethod
        def from_connection_string(cls, connection_string):
            instance = cls()
            instance.connection_string = connection_string
            return instance

        def get_container_client(self, container_name):
            return object()

    monkeypatch.setattr(
        "core.providers.file.azure_blob.BlobServiceClient",
        FakeBlobServiceClient,
    )

    config = FileConfig(
        provider="azure_blob",
        container_name="documents",
        connection_string="UseDevelopmentStorage=true",
    )

    provider = R2RProviderFactory.create_file_provider(config)
    assert provider.__class__.__name__ == "AzureBlobFileProvider"


def test_azure_blob_zip_export_requires_document_ids(monkeypatch):
    class FakeBlobServiceClient:
        connection_string: str

        def __init__(self):
            self.connection_string = ""

        @classmethod
        def from_connection_string(cls, connection_string):
            instance = cls()
            instance.connection_string = connection_string
            return instance

        def get_container_client(self, container_name):
            return object()

    monkeypatch.setattr(
        "core.providers.file.azure_blob.BlobServiceClient",
        FakeBlobServiceClient,
    )

    config = FileConfig(
        provider="azure_blob",
        container_name="documents",
        connection_string="UseDevelopmentStorage=true",
    )

    provider = R2RProviderFactory.create_file_provider(config)

    with pytest.raises(R2RException, match="Document IDs must be provided"):
        asyncio.run(provider.retrieve_files_as_zip())
