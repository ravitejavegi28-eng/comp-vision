"""SQLite persistence with atomic data validation and model activation."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "inspectra.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS products (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, source TEXT NOT NULL,
                    active_model TEXT, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS samples (
                    id TEXT NOT NULL, product_id TEXT NOT NULL REFERENCES products(id),
                    name TEXT NOT NULL, path TEXT NOT NULL, label TEXT NOT NULL,
                    split TEXT NOT NULL, session_id TEXT NOT NULL, source TEXT NOT NULL,
                    digest TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY(product_id, id)
                );
                CREATE INDEX IF NOT EXISTS samples_product ON samples(product_id, split);
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(id),
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS inspections (
                    id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(id),
                    created_at TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS inspections_product ON inspections(product_id, created_at);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(id),
                    created_at TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reports (
                    product_id TEXT PRIMARY KEY REFERENCES products(id), payload TEXT NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def products(self) -> list[dict]:
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT id,name,source FROM products ORDER BY created_at,id")]

    def product(self, product_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM products WHERE id=?", (product_id,)).fetchone()
            return dict(row) if row else None

    def add_product(self, product_id: str, name: str, source: str) -> dict:
        with self.connect() as db:
            db.execute("INSERT INTO products(id,name,source,created_at) VALUES(?,?,?,?)", (product_id, name, source, now()))
        return {"id": product_id, "name": name, "source": source}

    def samples(self, product_id: str, split: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM samples WHERE product_id=?", [product_id]
        if split:
            sql += " AND split=?"
            args.append(split)
        with self.connect() as db:
            return [dict(row) for row in db.execute(sql + " ORDER BY created_at,id", args)]

    def sample(self, product_id: str, sample_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM samples WHERE product_id=? AND id=?", (product_id, sample_id)).fetchone()
            return dict(row) if row else None

    def add_samples(self, product_id: str, records: list[dict], *, seed: bool = False) -> None:
        """Validate and insert an entire batch atomically, including leakage checks."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for record in records:
                if record["split"] == "train" and record["label"] != "good":
                    raise ValueError("Training accepts good products only.")
                conflicts = db.execute(
                    "SELECT split FROM samples WHERE product_id=? AND session_id=? AND split<>? LIMIT 1",
                    (product_id, record["session_id"], record["split"]),
                ).fetchone()
                if conflicts:
                    raise ValueError(f"Session '{record['session_id']}' already belongs to {conflicts['split']}; keep each capture session in one split.")
                duplicate = db.execute(
                    "SELECT split,label FROM samples WHERE product_id=? AND digest=? LIMIT 1", (product_id, record["digest"])
                ).fetchone()
                if duplicate:
                    if duplicate["split"] != record["split"]:
                        raise ValueError(f"Image '{record['name']}' already exists in {duplicate['split']}; duplicates cannot cross splits.")
                    if not seed:
                        raise ValueError(f"Image '{record['name']}' is already in this dataset.")
                db.execute(
                    "INSERT OR IGNORE INTO samples(id,product_id,name,path,label,split,session_id,source,digest,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (record["id"], product_id, record["name"], str(Path(record["path"]).resolve()), record["label"], record["split"], record["session_id"], record["source"], record["digest"], now()),
                )
            sources = [row[0] for row in db.execute("SELECT DISTINCT source FROM samples WHERE product_id=?", (product_id,))]
            if sources:
                db.execute("UPDATE products SET source=? WHERE id=?", (sources[0] if len(sources) == 1 else "mixed", product_id))

    def dataset_counts(self, product_id: str) -> dict:
        counts = {"train_good": 0, "validation_good": 0, "validation_defect": 0, "test_good": 0, "test_defect": 0}
        with self.connect() as db:
            for row in db.execute("SELECT split,label,count(*) AS n FROM samples WHERE product_id=? GROUP BY split,label", (product_id,)):
                counts[f"{row['split']}_{row['label']}"] = row["n"]
        return counts

    def active_model(self, product_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT models.payload FROM models JOIN products ON products.active_model=models.id WHERE products.id=?", (product_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def activate_model(self, product_id: str, model: dict) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO models(id,product_id,payload) VALUES(?,?,?)", (model["id"], product_id, json.dumps(model)))
            db.execute("UPDATE products SET active_model=? WHERE id=?", (model["id"], product_id))
            db.execute("DELETE FROM reports WHERE product_id=?", (product_id,))

    def save_inspection(self, product_id: str, inspection: dict) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO inspections(id,product_id,created_at,payload) VALUES(?,?,?,?)", (inspection["id"], product_id, inspection["created_at"], json.dumps(inspection)))

    def inspections(self, product_id: str, limit: int | None = None) -> list[dict]:
        sql, args = "SELECT payload FROM inspections WHERE product_id=? ORDER BY created_at DESC", [product_id]
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(sql, args)]

    def review_inspection(self, product_id: str, inspection_id: str, review: dict) -> dict | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM inspections WHERE product_id=? AND id=?", (product_id, inspection_id)).fetchone()
            if not row:
                return None
            item = json.loads(row[0])
            item["review"] = review
            db.execute("UPDATE inspections SET payload=? WHERE product_id=? AND id=?", (json.dumps(item), product_id, inspection_id))
            return item

    def stats(self, product_id: str) -> dict:
        counts = {"total": 0, "pass": 0, "review": 0, "fail": 0, "recapture": 0}
        with self.connect() as db:
            rows = db.execute("SELECT json_extract(payload,'$.decision') AS decision,COUNT(*) AS n FROM inspections WHERE product_id=? GROUP BY decision", (product_id,))
            for row in rows:
                counts[row["decision"]] = row["n"]
                counts["total"] += row["n"]
        return counts

    def save_job(self, product_id: str, job: dict) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO jobs(id,product_id,created_at,payload) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", (job["id"], product_id, job["created_at"], json.dumps(job)))

    def job(self, product_id: str, job_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM jobs WHERE product_id=? AND id=?", (product_id, job_id)).fetchone()
            return json.loads(row[0]) if row else None

    def latest_job(self, product_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM jobs WHERE product_id=? ORDER BY created_at DESC LIMIT 1", (product_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def fail_interrupted_jobs(self) -> None:
        with self.connect() as db:
            rows = db.execute("SELECT id,payload FROM jobs").fetchall()
            for row in rows:
                job = json.loads(row["payload"])
                if job["status"] in ("queued", "running"):
                    job.update(status="failed", message="Interrupted by application shutdown. Start the job again.", error="Application restarted before this job completed.", finished_at=now())
                    db.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(job), row["id"]))

    def save_report(self, product_id: str, report: dict) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO reports(product_id,payload) VALUES(?,?) ON CONFLICT(product_id) DO UPDATE SET payload=excluded.payload", (product_id, json.dumps(report)))

    def report(self, product_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM reports WHERE product_id=?", (product_id,)).fetchone()
            return json.loads(row[0]) if row else None
