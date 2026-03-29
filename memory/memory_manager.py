import sqlite3
import os
import json
from datetime import datetime

class MemoryManager:
    def __init__(self, db_path="d:\\file\\vscode\\llm_brain\\memory\\memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS interactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conv_id TEXT,
                        timestamp TEXT,
                        domain_label TEXT,
                        keywords TEXT,
                        summary TEXT,
                        raw_input TEXT,
                        raw_output TEXT,
                        large_file_path TEXT,
                        weight INTEGER DEFAULT 1
                    )""")
        conn.commit()
        conn.close()

    def add_memory(self, conv_id: str, domain_label: str, keywords: list, summary: str, 
                   raw_input: str, raw_output: str, large_file_path: str = ""):
        timestamp = datetime.now().isoformat()
        keywords_str = json.dumps(keywords)
        
        # Heavy/Super Large data rule logic (Text longer than 1000 characters could be heavy)
        if len(raw_input) > 5000:
            large_file_path = f"d:\\file\\vscode\\llm_brain\\memory\\backup_{timestamp.replace(':', '-')}.txt"
            with open(large_file_path, "w", encoding="utf-8") as f:
                f.write(raw_input)
            raw_input = "[Saved to external file]"

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO interactions (conv_id, timestamp, domain_label, keywords, summary, raw_input, raw_output, large_file_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (conv_id, timestamp, domain_label, keywords_str, summary, raw_input, raw_output, large_file_path))
        conn.commit()
        conn.close()

    def retrieve_memory(self, threshold=1, match_keywords=None):
        """Two-stage read: First read keywords and summary."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        query = "SELECT id, summary, keywords, weight FROM interactions WHERE weight >= ?"
        c.execute(query, (threshold,))
        results = c.fetchall()
        
        # Increment weight for matched memories
        for r in results:
            if match_keywords and any(kw in r[2] for kw in match_keywords):
                c.execute("UPDATE interactions SET weight = weight + 1 WHERE id = ?", (r[0],))
        conn.commit()
        conn.close()
        return [{"id": r[0], "summary": r[1], "keywords": json.loads(r[2]), "weight": r[3]} for r in results]

    def load_full_memory(self, memory_id: int):
        """Second-stage: Load full details if summary is not enough."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT raw_input, raw_output, large_file_path FROM interactions WHERE id = ?", (memory_id,))
        result = c.fetchone()
        conn.close()
        
        raw_in, raw_out, large_file = result
        if large_file and os.path.exists(large_file):
            with open(large_file, "r", encoding="utf-8") as f:
                raw_in = f.read()
        return {"input": raw_in, "output": raw_out}
