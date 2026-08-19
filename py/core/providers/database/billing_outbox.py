from datetime import datetime
from typing import Any
from uuid import UUID

from core.base import Handler
from core.billing import InterloomBillingContext

from .base import PostgresConnectionManager


class PostgresBillingOutboxHandler(Handler):
    TABLE_NAME = "interloom_billing_outbox"

    def __init__(
        self, project_name: str, connection_manager: PostgresConnectionManager
    ):
        super().__init__(project_name, connection_manager)

    async def create_tables(self) -> None:
        table = self._get_table_name(self.TABLE_NAME)
        await self.connection_manager.execute_query(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                event_id UUID PRIMARY KEY,
                ingestion_id UUID NOT NULL,
                organization_id UUID NOT NULL,
                file_id UUID,
                note_id UUID,
                operation TEXT NOT NULL,
                requested_model TEXT NOT NULL,
                resolved_model TEXT,
                provider_request_id TEXT,
                status TEXT NOT NULL,
                started_at TIMESTAMPTZ NOT NULL,
                ended_at TIMESTAMPTZ,
                input_tokens BIGINT,
                output_tokens BIGINT,
                cached_input_tokens BIGINT,
                error TEXT,
                delivery_attempts INTEGER NOT NULL DEFAULT 0,
                next_delivery_at TIMESTAMPTZ,
                delivered_at TIMESTAMPTZ,
                last_delivery_error TEXT,
                CHECK ((file_id IS NOT NULL)::int + (note_id IS NOT NULL)::int = 1),
                CHECK (status IN ('pending', 'completed', 'failed'))
            );
            CREATE INDEX IF NOT EXISTS idx_{self.project_name}_{self.TABLE_NAME}_delivery
                ON {table} (next_delivery_at, started_at)
                WHERE status = 'completed' AND delivered_at IS NULL;
            CREATE INDEX IF NOT EXISTS idx_{self.project_name}_{self.TABLE_NAME}_ingestion
                ON {table} (ingestion_id);
            """
        )

    async def create_pending(
        self,
        *,
        event_id: UUID,
        context: InterloomBillingContext,
        operation: str,
        requested_model: str,
        started_at: datetime,
    ) -> None:
        await self.connection_manager.execute_query(
            f"""
            INSERT INTO {self._get_table_name(self.TABLE_NAME)} (
                event_id, ingestion_id, organization_id, file_id, note_id,
                operation, requested_model, status, started_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, 'pending', $8)
            """,
            [
                event_id,
                context.ingestion_id,
                context.organization_id,
                context.file_id,
                context.note_id,
                operation,
                requested_model,
                started_at,
            ],
        )

    async def complete(
        self,
        *,
        event_id: UUID,
        provider_request_id: str | None,
        resolved_model: str | None,
        ended_at: datetime,
        input_tokens: int | None,
        output_tokens: int | None,
        cached_input_tokens: int | None,
    ) -> None:
        await self.connection_manager.execute_query(
            f"""
            UPDATE {self._get_table_name(self.TABLE_NAME)}
            SET status = 'completed', provider_request_id = $2,
                resolved_model = $3, ended_at = $4, input_tokens = $5,
                output_tokens = $6, cached_input_tokens = $7,
                next_delivery_at = NOW()
            WHERE event_id = $1 AND status = 'pending'
            """,
            [
                event_id,
                provider_request_id,
                resolved_model,
                ended_at,
                input_tokens,
                output_tokens,
                cached_input_tokens,
            ],
        )

    async def fail(
        self,
        *,
        event_id: UUID,
        ended_at: datetime,
        error: str,
    ) -> None:
        await self.connection_manager.execute_query(
            f"""
            UPDATE {self._get_table_name(self.TABLE_NAME)}
            SET status = 'failed', ended_at = $2, error = $3
            WHERE event_id = $1 AND status = 'pending'
            """,
            [event_id, ended_at, error],
        )

    async def claim_for_delivery(self, *, limit: int) -> list[dict[str, Any]]:
        rows = await self.connection_manager.fetch_query(
            f"""
            WITH claimed AS (
                SELECT event_id
                FROM {self._get_table_name(self.TABLE_NAME)}
                WHERE status = 'completed' AND delivered_at IS NULL
                    AND next_delivery_at <= NOW()
                ORDER BY next_delivery_at, started_at
                FOR UPDATE SKIP LOCKED
                LIMIT $1
            )
            UPDATE {self._get_table_name(self.TABLE_NAME)} AS outbox
            SET delivery_attempts = delivery_attempts + 1,
                next_delivery_at = NOW() + INTERVAL '1 minute'
            FROM claimed
            WHERE outbox.event_id = claimed.event_id
            RETURNING outbox.*
            """,
            [limit],
        )
        return [self._serialize_event(dict(row)) for row in rows]

    async def mark_delivered(self, *, event_id: UUID | str) -> None:
        await self.connection_manager.execute_query(
            f"""
            UPDATE {self._get_table_name(self.TABLE_NAME)}
            SET delivered_at = NOW(), last_delivery_error = NULL
            WHERE event_id = $1
            """,
            [UUID(str(event_id))],
        )

    async def mark_delivery_failed(
        self, *, event_id: UUID | str, error: str
    ) -> None:
        await self.connection_manager.execute_query(
            f"""
            UPDATE {self._get_table_name(self.TABLE_NAME)}
            SET last_delivery_error = $2,
                next_delivery_at = NOW()
                    + LEAST(POWER(2, delivery_attempts), 3600)
                    * INTERVAL '1 second'
            WHERE event_id = $1
            """,
            [UUID(str(event_id)), error],
        )

    @staticmethod
    def _serialize_event(event: dict[str, Any]) -> dict[str, Any]:
        serialized = {
            key: value.isoformat()
            if isinstance(value, datetime)
            else str(value)
            if isinstance(value, UUID)
            else value
            for key, value in event.items()
            if key
            not in {
                "delivery_attempts",
                "next_delivery_at",
                "delivered_at",
                "last_delivery_error",
            }
        }
        serialized["schema_version"] = 1
        serialized["billing_event_id"] = serialized.pop("event_id")
        return serialized
