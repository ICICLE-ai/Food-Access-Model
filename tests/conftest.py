"""Shared test fixtures.

``routes.py`` fails to import if any of DB_USER / DB_HOST / DB_PORT / DB_NAME
are missing, because it builds ``DATABASE_URL`` at module scope. The tests
never touch a real database (they stub `routes.pool`), so defaulting them to
placeholders here lets the module import under pytest without the developer
having to populate .env first.
"""
import os

os.environ.setdefault("DB_USER", "test-user")
os.environ.setdefault("DB_PASS", "test-pass")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_NAME", "test-db")
