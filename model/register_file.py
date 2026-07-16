
import sqlite3
import hashlib
from datetime import datetime
from model.config import *
DB_PATH = "file_uploads.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS uploaded_files (
            file_hash TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            saved_path TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
        )
        """
    )
    conn.commit()
    conn.close()


def hash_filename(file_name: str) -> str:
    return hashlib.sha256(file_name.encode("utf-8")).hexdigest()


def file_hash_exists(file_hash: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT 1 FROM uploaded_files WHERE file_hash = ? LIMIT 1", (file_hash,)
    )
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def register_file(file_hash: str, file_name: str, saved_path: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO uploaded_files (file_hash, file_name, saved_path, uploaded_at,status) "
        "VALUES (?, ?, ?, ?, ?)",
        (file_hash, file_name, saved_path, datetime.now().isoformat(timespec="seconds"),""),
    )
    conn.commit()
    conn.close()


def update_status(file_hash: str, status: str):
    if status not in FILE_STATUSES:
        raise ValueError(
            f"Status '{status}' không hợp lệ. Phải là một trong {FILE_STATUSES}"
        )

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE uploaded_files SET status = ? WHERE file_hash = ?",
        (status, file_hash),
    )
    conn.commit()
    conn.close()


