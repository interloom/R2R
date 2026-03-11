import json
import logging
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Optional
from uuid import UUID

from core.base import FileConfig, FileProvider, R2RException

logger = logging.getLogger()


class LocalFileProvider(FileProvider):
    """Local filesystem implementation of the FileProvider."""

    def __init__(self, config: FileConfig):
        super().__init__(config)
        base_path = self.config.base_path or os.getenv(
            "LOCAL_FILE_STORAGE_PATH"
        )
        if not base_path:
            raise ValueError(
                "Local file storage base path is required when using local provider"
            )
        self.base_path = Path(base_path)
        self.documents_path = self.base_path / "documents"

    def _document_path(self, document_id: UUID) -> Path:
        return self.documents_path / str(document_id)

    def _content_path(self, document_id: UUID) -> Path:
        return self._document_path(document_id) / "content"

    def _metadata_path(self, document_id: UUID) -> Path:
        return self._document_path(document_id) / "metadata.json"

    def _file_key(self, document_id: UUID) -> str:
        return str(Path("documents") / str(document_id) / "content")

    def _normalize_datetime(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _load_metadata(self, document_id: UUID) -> dict[str, Any]:
        metadata_path = self._metadata_path(document_id)
        if not metadata_path.exists():
            raise R2RException(
                status_code=404,
                message=f"File for document {document_id} not found",
            )

        with metadata_path.open(encoding="utf-8") as metadata_file:
            return json.load(metadata_file)

    def _write_file_atomically(
        self, destination: Path, payload: bytes
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", delete=False, dir=destination.parent
            ) as temp_file:
                temp_file.write(payload)
                temp_path = temp_file.name
            os.replace(temp_path, destination)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def _iter_metadata_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not self.documents_path.exists():
            return records

        for metadata_path in self.documents_path.glob("*/metadata.json"):
            try:
                with metadata_path.open(encoding="utf-8") as metadata_file:
                    records.append(json.load(metadata_file))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Skipping unreadable local file metadata %s: %s",
                    metadata_path,
                    exc,
                )

        return records

    async def initialize(self) -> None:
        self.documents_path.mkdir(parents=True, exist_ok=True)

    async def store_file(
        self,
        document_id: UUID,
        file_name: str,
        file_content: BinaryIO,
        file_type: Optional[str] = None,
    ) -> None:
        file_content.seek(0)
        payload = file_content.read()
        now = datetime.now(timezone.utc)

        document_path = self._document_path(document_id)
        existing_created_at: Optional[str] = None
        metadata_path = self._metadata_path(document_id)
        if metadata_path.exists():
            try:
                existing_created_at = self._load_metadata(document_id).get(
                    "created_at"
                )
            except R2RException:
                existing_created_at = None

        document_path.mkdir(parents=True, exist_ok=True)
        self._write_file_atomically(self._content_path(document_id), payload)

        metadata = {
            "document_id": str(document_id),
            "file_name": file_name,
            "file_type": file_type,
            "file_size": len(payload),
            "file_key": self._file_key(document_id),
            "created_at": existing_created_at or now.isoformat(),
            "updated_at": now.isoformat(),
        }
        self._write_file_atomically(
            self._metadata_path(document_id),
            json.dumps(metadata, sort_keys=True).encode("utf-8"),
        )

    async def retrieve_file(
        self, document_id: UUID
    ) -> Optional[tuple[str, BinaryIO, int]]:
        metadata = self._load_metadata(document_id)
        content_path = self._content_path(document_id)
        if not content_path.exists():
            raise R2RException(
                status_code=404,
                message=f"File for document {document_id} not found",
            )

        return (
            metadata["file_name"],
            BytesIO(content_path.read_bytes()),
            int(metadata["file_size"]),
        )

    async def retrieve_files_as_zip(
        self,
        document_ids: Optional[list[UUID]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> tuple[str, BinaryIO, int]:
        selected_ids: list[UUID]

        if document_ids:
            selected_ids = document_ids
        else:
            normalized_start = (
                self._normalize_datetime(start_date) if start_date else None
            )
            normalized_end = (
                self._normalize_datetime(end_date) if end_date else None
            )
            selected_ids = []
            for record in self._iter_metadata_records():
                created_at_raw = record.get("created_at")
                if not created_at_raw:
                    continue
                created_at = self._normalize_datetime(
                    datetime.fromisoformat(created_at_raw)
                )
                if normalized_start and created_at < normalized_start:
                    continue
                if normalized_end and created_at > normalized_end:
                    continue
                selected_ids.append(UUID(record["document_id"]))

        zip_buffer = BytesIO()
        with zipfile.ZipFile(
            zip_buffer, "w", zipfile.ZIP_DEFLATED
        ) as zip_file:
            for document_id in selected_ids:
                try:
                    result = await self.retrieve_file(document_id)
                except R2RException as exc:
                    if exc.status_code == 404:
                        continue
                    raise
                if not result:
                    continue
                file_name, file_content, _ = result
                file_content.seek(0)
                zip_file.writestr(file_name, file_content.read())

        zip_buffer.seek(0)
        zip_size = zip_buffer.getbuffer().nbytes
        if zip_size == 0:
            raise R2RException(
                status_code=404,
                message="No files found matching the specified criteria",
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"files_export_{timestamp}.zip"
        return zip_filename, zip_buffer, zip_size

    async def delete_file(self, document_id: UUID) -> bool:
        document_path = self._document_path(document_id)
        content_path = self._content_path(document_id)
        metadata_path = self._metadata_path(document_id)

        if not content_path.exists() and not metadata_path.exists():
            raise R2RException(
                status_code=404,
                message=f"File for document {document_id} not found",
            )

        if content_path.exists():
            content_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        if document_path.exists():
            try:
                document_path.rmdir()
            except OSError:
                logger.warning(
                    "Local document directory %s was not empty during delete",
                    document_path,
                )

        return True

    async def get_files_overview(
        self,
        offset: int,
        limit: int,
        filter_document_ids: Optional[list[UUID]] = None,
        filter_file_names: Optional[list[str]] = None,
    ) -> list[dict]:
        document_id_filter = (
            {str(document_id) for document_id in filter_document_ids}
            if filter_document_ids
            else None
        )
        results: list[dict[str, Any]] = []

        for record in self._iter_metadata_records():
            if (
                document_id_filter
                and record["document_id"] not in document_id_filter
            ):
                continue
            if (
                filter_file_names
                and record["file_name"] not in filter_file_names
            ):
                continue

            results.append(
                {
                    "document_id": UUID(record["document_id"]),
                    "file_name": record["file_name"],
                    "file_key": record["file_key"],
                    "file_size": record["file_size"],
                    "file_type": record.get("file_type"),
                    "created_at": datetime.fromisoformat(record["created_at"]),
                    "updated_at": datetime.fromisoformat(record["updated_at"]),
                }
            )

        results.sort(key=lambda row: row["created_at"], reverse=True)
        paginated = (
            results[offset:] if limit < 0 else results[offset : offset + limit]
        )

        if not paginated:
            raise R2RException(
                status_code=404,
                message="No files found with the given filters",
            )

        return paginated
