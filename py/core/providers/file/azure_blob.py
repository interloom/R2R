import logging
import os
import zipfile
from datetime import datetime
from io import BytesIO
from typing import BinaryIO, Optional
from uuid import UUID

from core.base import FileConfig, FileProvider, R2RException

try:
    from azure.core.exceptions import (
        ResourceExistsError,
        ResourceNotFoundError,
    )
    from azure.storage.blob import BlobServiceClient, ContentSettings
except ImportError:  # pragma: no cover - handled at runtime
    BlobServiceClient = None
    ContentSettings = None
    ResourceExistsError = ResourceNotFoundError = Exception

logger = logging.getLogger()


class AzureBlobFileProvider(FileProvider):
    """Azure Blob Storage implementation of the FileProvider.

    Like the S3 provider, object-store listing and export flows are optimized
    for explicit document IDs rather than broad database-style filtering.
    """

    def __init__(self, config: FileConfig):
        super().__init__(config)

        if BlobServiceClient is None:
            raise ImportError(
                "azure-storage-blob is required to use the Azure Blob file provider"
            )

        container_name = self.config.container_name or os.getenv(
            "AZURE_STORAGE_CONTAINER_NAME"
        )
        connection_string = self.config.connection_string or os.getenv(
            "AZURE_STORAGE_CONNECTION_STRING"
        )
        content_settings_cls = ContentSettings
        assert container_name is not None
        assert connection_string is not None
        assert content_settings_cls is not None

        self.container_name = container_name
        self.content_settings_cls = content_settings_cls

        self.blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        self.container_client = self.blob_service_client.get_container_client(
            self.container_name
        )

    def _get_blob_name(self, document_id: UUID) -> str:
        return f"documents/{document_id}"

    async def initialize(self) -> None:
        try:
            self.container_client.create_container()
            logger.info(
                f"Created Azure Blob container: {self.container_name}"
            )
        except ResourceExistsError:
            logger.info(
                f"Using existing Azure Blob container: {self.container_name}"
            )
        except Exception as e:
            logger.error(f"Error accessing Azure Blob container: {e}")
            raise R2RException(
                status_code=500,
                message=f"Failed to initialize Azure Blob container: {e}",
            ) from e

    async def store_file(
        self,
        document_id: UUID,
        file_name: str,
        file_content: BinaryIO,
        file_type: Optional[str] = None,
    ) -> None:
        try:
            blob_name = self._get_blob_name(document_id)
            blob_client = self.container_client.get_blob_client(blob_name)
            content_settings_cls = self.content_settings_cls

            file_content.seek(0)
            blob_client.upload_blob(
                file_content,
                overwrite=True,
                content_settings=content_settings_cls(
                    content_type=file_type or "application/octet-stream"
                ),
                metadata={
                    "filename": file_name,
                    "document_id": str(document_id),
                },
            )
        except Exception as e:
            logger.error(f"Error storing file in Azure Blob: {e}")
            raise R2RException(
                status_code=500,
                message=f"Failed to store file in Azure Blob: {e}",
            ) from e

    async def retrieve_file(
        self, document_id: UUID
    ) -> Optional[tuple[str, BinaryIO, int]]:
        blob_name = self._get_blob_name(document_id)
        blob_client = self.container_client.get_blob_client(blob_name)

        try:
            properties = blob_client.get_blob_properties()
            downloader = blob_client.download_blob()
            file_content = BytesIO(downloader.readall())
            file_content.seek(0)

            metadata = properties.metadata or {}
            file_name = metadata.get("filename", f"file-{document_id}")
            file_size = properties.size or 0

            return file_name, file_content, file_size
        except ResourceNotFoundError as e:
            raise R2RException(
                status_code=404,
                message=f"File for document {document_id} not found",
            ) from e
        except Exception as e:
            raise R2RException(
                status_code=500,
                message=f"Error retrieving file from Azure Blob: {e}",
            ) from e

    async def retrieve_files_as_zip(
        self,
        document_ids: Optional[list[UUID]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> tuple[str, BinaryIO, int]:
        """Build a zip from explicitly requested document IDs.

        Azure Blob mirrors the S3 provider here: broad object-store exports are
        intentionally not supported in the same way as Postgres-backed storage.
        """
        if not document_ids:
            raise R2RException(
                status_code=400,
                message="Document IDs must be provided for Azure Blob file retrieval",
            )

        zip_buffer = BytesIO()

        with zipfile.ZipFile(
            zip_buffer, "w", zipfile.ZIP_DEFLATED
        ) as zip_file:
            for doc_id in document_ids:
                try:
                    result = await self.retrieve_file(doc_id)
                    if not result:
                        continue

                    file_name, file_content, _ = result
                    file_content.seek(0)
                    content_bytes = file_content.read()

                    zip_file.writestr(file_name, content_bytes)
                except R2RException as e:
                    if e.status_code == 404:
                        logger.warning(
                            f"File for document {doc_id} not found, skipping"
                        )
                        continue
                    raise

        zip_buffer.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"files_export_{timestamp}.zip"
        zip_size = zip_buffer.getbuffer().nbytes

        if zip_size == 0:
            raise R2RException(
                status_code=404,
                message="No files found for the specified document IDs",
            )

        return zip_filename, zip_buffer, zip_size

    async def delete_file(self, document_id: UUID) -> bool:
        blob_name = self._get_blob_name(document_id)
        blob_client = self.container_client.get_blob_client(blob_name)

        try:
            blob_client.delete_blob()
            return True
        except ResourceNotFoundError as e:
            raise R2RException(
                status_code=404,
                message=f"File for document {document_id} not found",
            ) from e
        except Exception as e:
            logger.error(f"Error deleting file from Azure Blob: {e}")
            raise R2RException(
                status_code=500,
                message=f"Failed to delete file from Azure Blob: {e}",
            ) from e

    async def get_files_overview(
        self,
        offset: int,
        limit: int,
        filter_document_ids: Optional[list[UUID]] = None,
        filter_file_names: Optional[list[str]] = None,
    ) -> list[dict]:
        """Get an overview of stored files.

        This behaves like the S3 provider and works best when callers provide
        explicit document IDs.
        """
        results = []

        if filter_document_ids:
            for doc_id in filter_document_ids:
                blob_name = self._get_blob_name(doc_id)
                blob_client = self.container_client.get_blob_client(blob_name)

                try:
                    properties = blob_client.get_blob_properties()
                    file_name = (properties.metadata or {}).get(
                        "filename", f"file-{doc_id}"
                    )

                    if filter_file_names and file_name not in filter_file_names:
                        continue

                    results.append(
                        {
                            "document_id": doc_id,
                            "file_name": file_name,
                            "file_key": blob_name,
                            "file_size": properties.size or 0,
                            "file_type": properties.content_settings.content_type,
                            "created_at": properties.creation_time,
                            "updated_at": properties.last_modified,
                        }
                    )
                except ResourceNotFoundError:
                    continue
        else:
            try:
                blobs = self.container_client.list_blobs(
                    name_starts_with="documents/"
                )
                page_items = list(blobs)[offset : offset + limit]

                for item in page_items:
                    doc_id_str = item.name.split("/")[-1]
                    try:
                        doc_id = UUID(doc_id_str)
                    except ValueError:
                        continue

                    blob_client = self.container_client.get_blob_client(
                        item.name
                    )
                    properties = blob_client.get_blob_properties()
                    file_name = (properties.metadata or {}).get(
                        "filename", f"file-{doc_id}"
                    )

                    if filter_file_names and file_name not in filter_file_names:
                        continue

                    results.append(
                        {
                            "document_id": doc_id,
                            "file_name": file_name,
                            "file_key": item.name,
                            "file_size": item.size,
                            "file_type": properties.content_settings.content_type,
                            "created_at": properties.creation_time,
                            "updated_at": properties.last_modified,
                        }
                    )
            except Exception as e:
                logger.error(f"Error listing files in Azure Blob: {e}")
                raise R2RException(
                    status_code=500,
                    message=f"Failed to list files from Azure Blob: {e}",
                ) from e

        if not results:
            raise R2RException(
                status_code=404,
                message="No files found with the given filters",
            )

        return results
