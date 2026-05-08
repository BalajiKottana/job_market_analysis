"""
pipeline_watermark.py
─────────────────────
Shared watermark infrastructure for the job market pipeline.

Design decisions from conversation:
  - Org-agnostic: new scrapers call register_org() — zero changes here
  - stage column: normalize uses (org_key, stage), downstream uses (_pipeline, stage)
    so chunking/vectorization are never blocked by per-org sync issues
  - MERGE on all writes — idempotent, safe to re-run
  - last_processed_dt advances ONLY on mark_success() / mark_no_data()
  - mark_failed() leaves date unchanged → automatic retry next run
  - _reg_cache prevents redundant MERGE calls within same session

Import pattern (Databricks Repos — file is plain .py, not a notebook):
    from pipeline_watermark import WatermarkManager, WATERMARK_TABLE, CLEAN_TABLE
"""


import os
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from datetime import date
from typing import Optional
from properties import RAW_TABLE, WATERMARK_TABLE, CLEAN_TABLE, HISTORY_TABLE
  



# ─────────────────────────────────────────────────────────────
# WATERMARK MANAGER
# ─────────────────────────────────────────────────────────────

class WatermarkManager:
    """
    Manages the pipeline_watermarks table.

    Table schema (one row per org_key + stage):
        org_key           STRING   -- "google" | "amazon" | "_pipeline"
        stage             STRING   -- "normalize" | "chunk" | "vectorize"
        last_processed_dt DATE     -- advances only on success/no_data
        last_run_at       TIMESTAMP
        last_run_status   STRING   -- success | no_data | skipped | failed
        rows_processed    LONG
        notes             STRING

    Key rule: scrapers (01, 02) call register_org() only.
              03_data_cleaning calls mark_success/failed/no_data for normalize.
              04_chunker and 05_embedder use _pipeline stage.
    """

    def __init__(self, spark: SparkSession):
        self.spark      = spark
        self._reg_cache: set = set()   # skip redundant MERGEs in same session

    # ── REGISTRATION ─────────────────────────────────────────

    def register_org(self, org_key: str, stage: str = "normalize",
                     seed_dt: str = "2024-01-01"):
        """
        Registers an (org_key, stage) pair. Idempotent — MERGE does nothing
        if the row already exists. Cached in memory to avoid repeated Delta I/O
        within the same notebook session.

        Scrapers call: wm.register_org("google")
        Bootstrap:     wm.register_org("_pipeline", "chunk")
        """
        key = f"{org_key.lower()}::{stage.lower()}"
        if key in self._reg_cache:
            return

        self.spark.sql(f"""
            MERGE INTO {WATERMARK_TABLE} AS target
            USING (
                SELECT
                    '{org_key.lower()}'     AS org_key,
                    '{stage.lower()}'       AS stage,
                    DATE'{seed_dt}'         AS last_processed_dt,
                    current_timestamp()     AS last_run_at,
                    'success'               AS last_run_status,
                    0L                      AS rows_processed,
                    'initial registration'  AS notes
            ) AS source
            ON  target.org_key = source.org_key
            AND target.stage   = source.stage
            WHEN NOT MATCHED THEN INSERT *
        """)
        self._reg_cache.add(key)
        print(f"  Watermark registered: [{org_key}] stage={stage} seed={seed_dt}")

    def list_registered_orgs(self, stage: str = "normalize") -> list:
        """Returns all org_keys for a given stage (excludes _pipeline)."""
        return [
            r["org_key"]
            for r in self.spark.table(WATERMARK_TABLE)
                               .filter(
                                   (F.col("stage")   == stage.lower()) &
                                   (F.col("org_key") != "_pipeline")
                               )
                               .select("org_key")
                               .distinct()
                               .collect()
        ]

    # ── READ — per-org (normalize stage) ─────────────────────

    def get_last_processed(self, org_key: str,
                           stage: str = "normalize") -> Optional[date]:
        """
        Returns the most recent date where this org+stage ran successfully
        (status = success OR no_data — both advance the watermark).
        """
        rows = (
            self.spark.table(WATERMARK_TABLE)
            .filter(
                (F.lower(F.col("org_key")) == org_key.lower()) &
                (F.lower(F.col("stage"))   == stage.lower())   &
                (F.col("last_run_status").isin("success",'skipped' ,"no_data"))
            )
            .orderBy(F.col("last_processed_dt").desc())
            .limit(1)
            .collect()
        )
        return rows[0]["last_processed_dt"] if rows else None

    def get_new_dates(self, org_key: str) -> list:
        """
        Returns file_dt values in raw_openings for this org beyond the
        normalize watermark. Empty list = no new data.
        """
        last_dt = self.get_last_processed(org_key, stage="normalize")
        df = self.spark.table(RAW_TABLE).filter(
            F.lower(F.col("organization")) == org_key.lower()
        )
        if last_dt is not None:
            df = df.filter(F.col("file_dt") > F.lit(last_dt))
        return [
            r["file_dt"]
            for r in df.select("file_dt").distinct().orderBy("file_dt").collect()
        ]

    def _get_last_status(self, org_key: str, stage: str) -> str:
        rows = (
            self.spark.table(WATERMARK_TABLE)
            .filter(
                (F.lower(F.col("org_key")) == org_key.lower()) &
                (F.lower(F.col("stage"))   == stage.lower())
            )
            .orderBy(F.col("last_run_at").desc())
            .limit(1)
            .collect()
        )
        return rows[0]["last_run_status"] if rows else "unknown"

    def check_availability(self, orgs: list = None) -> dict:
        """
        Checks new-data availability for all registered normalize-stage orgs.

        Returns dict per org:
          { "ready": bool, "last_watermark": date,
            "new_dates": [...], "max_new_dt": date, "last_status": str }
        """
        if orgs is None:
            orgs = self.list_registered_orgs(stage="normalize")
        if not orgs:
            print("  No orgs registered.")
            return {}

        result = {}
        print("\n── Availability check ──────────────────────────────────")
        for org in orgs:
            new_dates   = self.get_new_dates(org)
            last_wm     = self.get_last_processed(org, stage="normalize")
            last_status = self._get_last_status(org, stage="normalize")
            ready       = len(new_dates) > 0

            print(f"new_dates={new_dates},last_wm={last_wm},last_status={last_status}")

            result[org] = {
                "ready":          ready,
                "last_watermark": last_wm,
                "new_dates":      new_dates,
                "max_new_dt":     max(new_dates) if new_dates else None,
                "last_status":    last_status,
            }
            status_str = "READY" if ready else "NO NEW DATA"
            print(
                f"  [{org.upper():<12}] {status_str} | "
                f"watermark={last_wm} | last_status={last_status} | "
                f"new_dates={[str(d) for d in new_dates]}"
            )
        print("────────────────────────────────────────────────────────\n")
        return result

    # ── READ — pipeline-level (chunk / vectorize stages) ─────

    def get_pipeline_watermark(self) -> Optional[date]:
        """
        Upper bound for chunking: MIN(last_processed_dt) across all orgs at
        normalize stage. Chunking can never get ahead of the slowest org.
        Returns None if any org has never successfully normalized.
        """
        rows = (
            self.spark.table(WATERMARK_TABLE)
            .filter(
                (F.col("stage")          == "normalize") &
                (F.col("org_key")        != "_pipeline") &
                (F.col("last_run_status").isin("success", "no_data"))
            )
            .agg(F.min("last_processed_dt").alias("min_dt"))
            .collect()
        )
        return rows[0]["min_dt"] if rows and rows[0]["min_dt"] else None

    def get_stage_watermark(self, stage: str) -> Optional[date]:
        """Returns last_processed_dt for a _pipeline-level stage."""
        return self.get_last_processed("_pipeline", stage=stage)

    # ── WRITE ────────────────────────────────────────────────

    def _write_history(self, org_key: str, stage: str, file_dt,
                       status: str, rows: int, notes: str):
        """Append one row to history — never updates, full audit trail."""
        safe_notes = str(notes or "").replace("'", "''")[:500]
        self.spark.sql(f"""
            INSERT INTO {HISTORY_TABLE}
            SELECT
                '{org_key.lower()}'  AS org_key,
                '{stage.lower()}'    AS stage,
                DATE'{file_dt}'      AS last_processed_dt,
                current_timestamp()  AS last_run_at,
                '{status}'           AS last_run_status,
                {int(rows)}L         AS rows_processed,
                '{safe_notes}'       AS notes
        """)

    def mark_success(self, org_key: str, max_file_dt: date,
                     rows_processed: int, stage: str = "normalize"):
        """Advances last_processed_dt. Call after full pipeline success."""
        self.spark.sql(f"""
            MERGE INTO {WATERMARK_TABLE} AS target
            USING (
                SELECT
                    '{org_key.lower()}'    AS org_key,
                    '{stage.lower()}'      AS stage,
                    DATE'{max_file_dt}'    AS last_processed_dt,
                    current_timestamp()    AS last_run_at,
                    'success'              AS last_run_status,
                    {int(rows_processed)}L AS rows_processed,
                    NULL                   AS notes
            ) AS source
            ON  target.org_key = source.org_key
            AND target.stage   = source.stage
            WHEN MATCHED     THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """)
        self._write_history(org_key, stage, max_file_dt, "success", rows_processed, "")
        print(f"  ✓ [{org_key}] stage={stage} → {max_file_dt} ({rows_processed} rows)")

    def mark_no_data(self, org_key: str, stage: str = "normalize"):
        """
        Scrape succeeded but found zero new rows.
        Advances date to today so pipeline does not wait indefinitely.
        """
        today = date.today()
        self.spark.sql(f"""
            MERGE INTO {WATERMARK_TABLE} AS target
            USING (
                SELECT
                    '{org_key.lower()}'                AS org_key,
                    '{stage.lower()}'                  AS stage,
                    DATE'{today}'                      AS last_processed_dt,
                    current_timestamp()                AS last_run_at,
                    'no_data'                          AS last_run_status,
                    0L                                 AS rows_processed,
                    'scrape succeeded, zero new rows'  AS notes
            ) AS source
            ON  target.org_key = source.org_key
            AND target.stage   = source.stage
            WHEN MATCHED     THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """)
        self._write_history(org_key, stage, today, "no_data", 0,
                            "scrape succeeded, zero new rows")
        print(f"  – [{org_key}] stage={stage}: no_data, watermark advanced to {today}")

    def mark_loaded(self, org_key: str, stage: str = "normalize"):
        """
        Scrape succeeded but found zero new rows.
        Advances date to today so pipeline does not wait indefinitely.
        """
        today = date.today()
        self.spark.sql(f"""
            MERGE INTO {WATERMARK_TABLE} AS target
            USING (
                SELECT
                    '{org_key.lower()}'                AS org_key,
                    '{stage.lower()}'                  AS stage,
                    DATE'{today}'                      AS last_processed_dt,
                    current_timestamp()                AS last_run_at,
                    'loaded'                          AS last_run_status,
                    0L                                 AS rows_processed,
                    'raw data loaded'  AS notes
            ) AS source
            ON  target.org_key = source.org_key
            AND target.stage   = source.stage
            WHEN MATCHED     THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """)
        self._write_history(org_key, stage, today, "no_data", 0,
                            "scrape succeeded, zero new rows")
        print(f"  – [{org_key}] stage={stage}: raw data loaded, watermark advanced to {today}")

    def mark_skipped(self, org_key: str, reason: str,
                     stage: str = "normalize"):
        """Records skip WITHOUT advancing last_processed_dt."""
        safe = reason.replace("'", "''")
        self.spark.sql(f"""
            MERGE INTO {WATERMARK_TABLE} AS target
            USING (
                SELECT
                    '{org_key.lower()}'  AS org_key,
                    '{stage.lower()}'    AS stage,
                    last_processed_dt,
                    current_timestamp()  AS last_run_at,
                    'skipped'            AS last_run_status,
                    0L                   AS rows_processed,
                    '{safe}'             AS notes
                FROM {WATERMARK_TABLE}
                WHERE org_key = '{org_key.lower()}'
                AND   stage   = '{stage.lower()}'
            ) AS source
            ON  target.org_key = source.org_key
            AND target.stage   = source.stage
            WHEN MATCHED THEN UPDATE SET
                last_run_at     = source.last_run_at,
                last_run_status = source.last_run_status,
                rows_processed  = source.rows_processed,
                notes           = source.notes
        """)
        print(f"  – [{org_key}] stage={stage}: skipped — {reason}")

    def mark_failed(self, org_key: str, reason: str,
                    stage: str = "normalize"):
        """
        Records failure WITHOUT advancing last_processed_dt.
        Next run automatically retries the same file_dt range.
        """
        safe = reason.replace("'", "''")[:500]
        self.spark.sql(f"""
            MERGE INTO {WATERMARK_TABLE} AS target
            USING (
                SELECT
                    '{org_key.lower()}'  AS org_key,
                    '{stage.lower()}'    AS stage,
                    last_processed_dt,
                    current_timestamp()  AS last_run_at,
                    'failed'             AS last_run_status,
                    0L                   AS rows_processed,
                    '{safe}'             AS notes
                FROM {WATERMARK_TABLE}
                WHERE org_key = '{org_key.lower()}'
                AND   stage   = '{stage.lower()}'
            ) AS source
            ON  target.org_key = source.org_key
            AND target.stage   = source.stage
            WHEN MATCHED THEN UPDATE SET
                last_run_at     = source.last_run_at,
                last_run_status = source.last_run_status,
                rows_processed  = source.rows_processed,
                notes           = source.notes
        """)
        self._write_history(org_key, stage,
                            self.get_last_processed(org_key, stage) or "1970-01-01",
                            "failed", 0, safe)
        print(f"  ✗ [{org_key}] stage={stage}: failed — {reason[:80]}")


    def check_if_rerun(self, org_key: str, stage: str = "normalize",
                    scrape_tried_dt: date = None) -> bool:
        try:
            already_scraped = (
                self.spark.table(RAW_TABLE)          # fix: self.spark
                .filter(
                    (F.lower(F.col("scrape_org_key")) == F.lit(org_key)) &
                    (F.col("file_dt") == F.lit(scrape_tried_dt))
                )
                .limit(1)
                .count()
            ) > 0

            if already_scraped:
                self.mark_skipped(org_key, f"file_dt={scrape_tried_dt} already present in raw_openings")

        except Exception as e:                       # fix: catch specific + print real error
            print(f"Error checking rerun for {org_key}: {e}")
            return False

        return already_scraped


    


    # ── PIPELINE-LEVEL STAGE METHODS ─────────────────────────

    def mark_stage_success(self, stage: str, max_file_dt: date, rows: int):
        """Advances a _pipeline-level stage watermark (chunk / vectorize)."""
        self.mark_success("_pipeline", max_file_dt, rows, stage=stage)

    def mark_stage_failed(self, stage: str, reason: str):
        """Records failure for a _pipeline-level stage."""
        self.mark_failed("_pipeline", reason, stage=stage)

    def status(self):
        """Print current watermark state."""
        print(f"\n{'═' * 65}")
        print(f"  pipeline_watermarks — {WATERMARK_TABLE}")
        print(f"{'═' * 65}")
        self.spark.table(WATERMARK_TABLE).orderBy("org_key", "stage").show(truncate=False)

