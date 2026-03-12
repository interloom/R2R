"""Database utility functions for PostgreSQL operations."""

import re


_REGCONFIG_NAME_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$"
)


def psql_quote_literal(value: str) -> str:
    """Safely quote a string literal for PostgreSQL to prevent SQL injection.

    This is a simple implementation - in production, you should use proper parameterization
    or your database driver's quoting functions.
    """
    return "'" + value.replace("'", "''") + "'"


def psql_regconfig_literal(value: str) -> str:
    """Return a validated PostgreSQL regconfig literal.

    Accepts standard text search configuration names like ``english`` or
    schema-qualified names like ``pg_catalog.english``.
    """
    if not _REGCONFIG_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            f"Invalid PostgreSQL text search configuration: {value}"
        )
    return f"{psql_quote_literal(value)}::regconfig"
