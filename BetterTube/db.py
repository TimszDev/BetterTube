from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "bettertube.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table_name});").fetchall()
    return any(str(row["name"]) == column_name for row in rows)


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS watched_videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL UNIQUE,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                creator TEXT NOT NULL,
                duration_seconds INTEGER NOT NULL DEFAULT 0,
                description TEXT NOT NULL DEFAULT '',
                context_text TEXT NOT NULL DEFAULT '',
                tags_json TEXT NOT NULL DEFAULT '[]',
                subjects_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                result_json TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL UNIQUE,
                url TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                creator TEXT NOT NULL DEFAULT '',
                duration_seconds INTEGER NOT NULL DEFAULT 0,
                description TEXT NOT NULL DEFAULT '',
                context_text TEXT NOT NULL DEFAULT '',
                tags_json TEXT NOT NULL DEFAULT '[]',
                subjects_json TEXT NOT NULL DEFAULT '[]',
                feedback INTEGER NOT NULL CHECK (feedback IN (-1, 1)),
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS active_suggestions (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                suggestions_json TEXT NOT NULL DEFAULT '[]',
                target_count INTEGER NOT NULL DEFAULT 3,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        if not _column_exists(conn, "watched_videos", "context_text"):
            conn.execute(
                "ALTER TABLE watched_videos ADD COLUMN context_text TEXT NOT NULL DEFAULT '';"
            )
        if not _column_exists(conn, "recommendation_feedback", "duration_seconds"):
            conn.execute(
                "ALTER TABLE recommendation_feedback ADD COLUMN duration_seconds INTEGER NOT NULL DEFAULT 0;"
            )
        if not _column_exists(conn, "recommendation_feedback", "description"):
            conn.execute(
                "ALTER TABLE recommendation_feedback ADD COLUMN description TEXT NOT NULL DEFAULT '';"
            )
        if not _column_exists(conn, "recommendation_feedback", "context_text"):
            conn.execute(
                "ALTER TABLE recommendation_feedback ADD COLUMN context_text TEXT NOT NULL DEFAULT '';"
            )
        if not _column_exists(conn, "recommendation_feedback", "tags_json"):
            conn.execute(
                "ALTER TABLE recommendation_feedback ADD COLUMN tags_json TEXT NOT NULL DEFAULT '[]';"
            )
        if not _column_exists(conn, "recommendation_feedback", "subjects_json"):
            conn.execute(
                "ALTER TABLE recommendation_feedback ADD COLUMN subjects_json TEXT NOT NULL DEFAULT '[]';"
            )
        conn.commit()


def add_watched_video(video: dict[str, Any]) -> bool:
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO watched_videos (
                video_id, url, title, creator, duration_seconds, description, context_text, tags_json, subjects_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                video["video_id"],
                video["url"],
                video["title"],
                video["creator"],
                int(video.get("duration_seconds") or 0),
                video.get("description") or "",
                video.get("context_text") or "",
                json.dumps(video.get("tags") or []),
                json.dumps(video.get("subjects") or []),
            ),
        )
        conn.commit()
        return cursor.rowcount > 0


def get_watched_videos() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                video_id,
                url,
                title,
                creator,
                duration_seconds,
                description,
                context_text,
                tags_json,
                subjects_json,
                created_at
            FROM watched_videos
            ORDER BY created_at DESC;
            """
        ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["tags"] = json.loads(item.pop("tags_json") or "[]")
        item["subjects"] = json.loads(item.pop("subjects_json") or "[]")
        results.append(item)
    return results


def list_trained_videos(limit: int = 20) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                title,
                creator,
                url,
                duration_seconds,
                created_at
            FROM watched_videos
            ORDER BY created_at DESC
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_video_count() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS total FROM watched_videos;").fetchone()
    return int(row["total"] if row else 0)


def get_watched_video_ids() -> set[str]:
    with _connect() as conn:
        rows = conn.execute("SELECT video_id FROM watched_videos;").fetchall()
    return {str(row["video_id"]) for row in rows}


def log_recommendations(results: list[dict[str, Any]]) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO recommendation_log (result_json) VALUES (?);",
            (json.dumps(results),),
        )
        conn.commit()


def set_recommendation_feedback(
    video_id: str,
    url: str,
    title: str,
    creator: str,
    feedback: int,
    duration_seconds: int = 0,
    description: str = "",
    context_text: str = "",
    tags: list[str] | None = None,
    subjects: list[str] | None = None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO recommendation_feedback (
                video_id, url, title, creator, duration_seconds, description, context_text, tags_json, subjects_json, feedback
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                url=excluded.url,
                title=excluded.title,
                creator=excluded.creator,
                duration_seconds=excluded.duration_seconds,
                description=excluded.description,
                context_text=excluded.context_text,
                tags_json=excluded.tags_json,
                subjects_json=excluded.subjects_json,
                feedback=excluded.feedback,
                updated_at=CURRENT_TIMESTAMP;
            """,
            (
                video_id,
                url,
                title,
                creator,
                int(duration_seconds or 0),
                description or "",
                context_text or "",
                json.dumps(tags or []),
                json.dumps(subjects or []),
                int(feedback),
            ),
        )
        conn.commit()


def get_recommendation_feedback_map() -> dict[str, int]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT video_id, feedback FROM recommendation_feedback;"
        ).fetchall()
    return {str(row["video_id"]): int(row["feedback"]) for row in rows}


def get_recommendation_feedback_entries() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                video_id,
                url,
                title,
                creator,
                duration_seconds,
                description,
                context_text,
                tags_json,
                subjects_json,
                feedback,
                updated_at
            FROM recommendation_feedback
            ORDER BY updated_at DESC;
            """
        ).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["tags"] = json.loads(item.pop("tags_json") or "[]")
        item["subjects"] = json.loads(item.pop("subjects_json") or "[]")
        results.append(item)
    return results


def clear_recommendations(clear_feedback: bool = False) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM recommendation_log;")
        conn.execute("DELETE FROM active_suggestions;")
        if clear_feedback:
            conn.execute("DELETE FROM recommendation_feedback;")
        conn.commit()


def set_active_suggestions(suggestions: list[dict[str, Any]], target_count: int) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO active_suggestions (id, suggestions_json, target_count)
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                suggestions_json=excluded.suggestions_json,
                target_count=excluded.target_count,
                updated_at=CURRENT_TIMESTAMP;
            """,
            (json.dumps(suggestions), int(target_count)),
        )
        conn.commit()


def get_active_suggestions() -> tuple[list[dict[str, Any]], int]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT suggestions_json, target_count FROM active_suggestions WHERE id = 1;"
        ).fetchone()
    if not row:
        return [], 3

    suggestions_json = row["suggestions_json"] or "[]"
    try:
        suggestions = json.loads(suggestions_json)
    except json.JSONDecodeError:
        suggestions = []
    target_count = int(row["target_count"] or 3)
    return suggestions, target_count


def clear_active_suggestions() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM active_suggestions;")
        conn.commit()


def clear_all_data() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM watched_videos;")
        conn.execute("DELETE FROM recommendation_log;")
        conn.execute("DELETE FROM recommendation_feedback;")
        conn.execute("DELETE FROM active_suggestions;")
        conn.commit()
