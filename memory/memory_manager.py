import sqlite3
import os
import json
from datetime import datetime
from typing import Optional
from config import config

class MemoryManager:
    def __init__(self, db_path: str = None, backup_dir: str = None):
        self.db_path = config.resolve_path(db_path or config.memory_db_path)
        self.backup_dir = config.resolve_path(backup_dir or config.memory_backup_dir)
        self._init_db()

    def _ensure_column(self, cursor, table_name: str, column_name: str, column_definition: str):
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = {row[1] for row in cursor.fetchall()}
        if column_name not in columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS interactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conv_id TEXT,
                        request_id TEXT,
                        timestamp TEXT,
                        domain_label TEXT,
                        keywords TEXT,
                        summary TEXT,
                        raw_input TEXT,
                        raw_output TEXT,
                        large_file_path TEXT,
                        weight INTEGER DEFAULT 1
                    )""")
        self._ensure_column(c, "interactions", "request_id", "TEXT")
        conn.commit()
        conn.close()

    def add_memory(self, conv_id: str, domain_label: str, keywords: list, summary: str,
                   raw_input: str, raw_output: str, large_file_path: str = "",
                   request_id: Optional[str] = None) -> int:
        timestamp = datetime.now().isoformat()
        keywords_str = json.dumps(keywords)
        
        # Heavy/Super Large data rule logic (Text longer than 1000 characters could be heavy)
        if len(raw_input) > 5000:
            large_file_path = os.path.join(self.backup_dir, f"backup_{timestamp.replace(':', '-')}.txt")
            with open(large_file_path, "w", encoding="utf-8") as f:
                f.write(raw_input)
            raw_input = "[Saved to external file]"

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO interactions (conv_id, request_id, timestamp, domain_label, keywords, summary, raw_input, raw_output, large_file_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (conv_id, request_id, timestamp, domain_label, keywords_str, summary, raw_input, raw_output, large_file_path),
        )
        memory_id = c.lastrowid
        conn.commit()
        conn.close()
        return memory_id

    def update_memory(self, memory_id: int, raw_output: Optional[str] = None,
                      summary: Optional[str] = None, keywords: Optional[list] = None,
                      raw_input: Optional[str] = None, request_id: Optional[str] = None):
        updates = []
        values = []

        if raw_output is not None:
            updates.append("raw_output = ?")
            values.append(raw_output)
        if summary is not None:
            updates.append("summary = ?")
            values.append(summary)
        if keywords is not None:
            updates.append("keywords = ?")
            values.append(json.dumps(keywords))
        if raw_input is not None:
            updates.append("raw_input = ?")
            values.append(raw_input)
        if request_id is not None:
            updates.append("request_id = ?")
            values.append(request_id)

        if not updates:
            return False

        values.append(memory_id)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(f"UPDATE interactions SET {', '.join(updates)} WHERE id = ?", values)
        conn.commit()
        updated = c.rowcount > 0
        conn.close()
        return updated

    def retrieve_memory(self, threshold=1, match_keywords=None, limit: int = 5,
                        exclude_conv_id: Optional[str] = None, exclude_ids: Optional[list] = None):
        """Two-stage read: First read keywords and summary."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        query = "SELECT id, conv_id, request_id, summary, keywords, weight FROM interactions WHERE weight >= ?"
        c.execute(query, (threshold,))
        results = c.fetchall()

        normalized_keywords = {
            kw.strip().lower() for kw in (match_keywords or [])
            if isinstance(kw, str) and kw.strip()
        }
        excluded_ids = set(exclude_ids or [])
        ranked_results = []

        for memory_id, conv_id, request_id, summary, keywords_raw, weight in results:
            if exclude_conv_id and conv_id == exclude_conv_id:
                continue
            if memory_id in excluded_ids:
                continue

            parsed_keywords = json.loads(keywords_raw)
            keyword_set = {
                kw.strip().lower() for kw in parsed_keywords
                if isinstance(kw, str) and kw.strip()
            }
            overlap_count = len(normalized_keywords & keyword_set)
            ranked_results.append({
                "id": memory_id,
                "request_id": request_id,
                "summary": summary,
                "keywords": parsed_keywords,
                "weight": weight,
                "overlap_count": overlap_count,
            })

        if normalized_keywords:
            matched_results = [item for item in ranked_results if item["overlap_count"] > 0]
            ranked_results = matched_results or ranked_results
            for item in matched_results:
                c.execute("UPDATE interactions SET weight = weight + 1 WHERE id = ?", (item["id"],))

        conn.commit()
        conn.close()

        if normalized_keywords:
            ranked_results.sort(key=lambda item: (item["overlap_count"], item["weight"], item["id"]), reverse=True)
        else:
            ranked_results.sort(key=lambda item: (item["weight"], item["id"]), reverse=True)
        return ranked_results[:limit]

    def load_full_memory(self, memory_id: int):
        """Second-stage: Load full details if summary is not enough."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT conv_id, request_id, raw_input, raw_output, large_file_path FROM interactions WHERE id = ?", (memory_id,))
        result = c.fetchone()
        conn.close()

        if not result:
            return None
        
        conv_id, request_id, raw_in, raw_out, large_file = result
        if large_file and os.path.exists(large_file):
            with open(large_file, "r", encoding="utf-8") as f:
                raw_in = f.read()
        return {"conv_id": conv_id, "request_id": request_id, "input": raw_in, "output": raw_out}
