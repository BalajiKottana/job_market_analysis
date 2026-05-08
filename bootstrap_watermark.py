import os
from pipeline_watermark import WatermarkManager

from properties import WATERMARK_TABLE, HISTORY_TABLE, catalog, schema_name,CHUNKS_TABLE

# ── Create watermark table ───────────────────────────────────

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {WATERMARK_TABLE} (
        org_key           STRING    NOT NULL
                          COMMENT 'Lowercase org key: google | amazon | _pipeline',
        stage             STRING    NOT NULL
                          COMMENT 'normalize | chunk | vectorize',
        last_processed_dt DATE
                          COMMENT 'Last file_dt successfully processed at this stage',
        last_run_at       TIMESTAMP COMMENT 'Wall-clock time of most recent run',
        last_run_status   STRING    COMMENT 'success | no_data | skipped | failed',
        rows_processed    LONG      COMMENT 'Rows written in that run',
        notes             STRING    COMMENT 'Skip reason, error snippet, or free-text'
    )
    USING DELTA
    COMMENT 'One row per (org_key, stage). Tracks pipeline high-water marks.
             last_processed_dt advances only on success or no_data.
             failed leaves it unchanged → automatic retry next run.'
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
print(f"Watermark table ready: {WATERMARK_TABLE}")

# COMMAND ----------
# ── Create history table (append-only audit log) ────────────

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {HISTORY_TABLE} (
        org_key           STRING,
        stage             STRING,
        last_processed_dt DATE,
        last_run_at       TIMESTAMP,
        last_run_status   STRING,
        rows_processed    LONG,
        notes             STRING
    )
    USING DELTA
    COMMENT 'Append-only audit trail. One row per run attempt.
             Query for: consecutive failures, no_data patterns, rows-per-week trends.'
    TBLPROPERTIES (delta.enableChangeDataFeed = false)
""")
print(f"History table ready: {HISTORY_TABLE}")

# COMMAND ----------
# ── Create clean_openings and quarantine tables ──────────────

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {catalog}.{schema_name}.clean_openings (
        doc_id                STRING,
        source_type           STRING,
        source_url            STRING,
        organization          STRING,
        org_key               STRING,
        job_id                STRING,
        title                 STRING,
        title_clean           STRING,
        job_location          STRING,
        job_location_clean    STRING,
        city                  STRING,
        country               STRING,
        job_position          STRING,
        job_description       STRING,
        job_family            STRING,
        seniority             STRING,
        seniority_source      STRING,
        team_name             STRING,
        domain                STRING,
        is_intern             BOOLEAN,
        trend_dt              DATE,
        data_quality_flags    ARRAY<STRING>,
        ingested_at           TIMESTAMP,
        file_dt               DATE,
        notional_job_posted_dt DATE
    )
    USING DELTA
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
print(f"clean_openings ready: {catalog}.{schema_name}.clean_openings")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {catalog}.{schema_name}.quarantine_openings (
        doc_id             STRING,
        organization       STRING,
        title              STRING,
        file_dt            DATE,
        quarantine_reason  STRING,
        quarantined_at     TIMESTAMP
    )
    USING DELTA
    TBLPROPERTIES (delta.enableChangeDataFeed = false)
""")
print(f"quarantine_openings ready: {catalog}.{schema_name}.quarantine_openings")


spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CHUNKS_TABLE} (
        chunk_id string, 
        sentence string ,
        window_text string ,
        section string ,
        section_type string ,
        sentence_index string ,
        doc_id string ,
        source_type string ,
        source_url string ,
        org_key string ,
        job_id string ,
        title_clean string ,
        job_family string ,
        domain string ,
        team_name string ,
        seniority string ,
        seniority_source string ,
        is_intern boolean ,
        job_location_clean string ,
        country string ,
        file_dt date ,
        notional_job_posted_dt date ,
        trend_dt date 
    )
    USING DELTA
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
print(f"job_sentance_chunks ready: {catalog}.{schema_name}.job_sentence_chunks")




# COMMAND ----------
# ── Seed watermark rows ──────────────────────────────────────

# wm = WatermarkManager(spark)

# # Per-org normalize stage
# wm.register_org("google",    stage="normalize", seed_dt="2026-03-25")
# wm.register_org("amazon",    stage="normalize", seed_dt="2026-03-25")
# # Add future orgs here: wm.register_org("microsoft", stage="normalize")

# # Pipeline-level downstream stages (org-agnostic)
# wm.register_org("_pipeline", stage="chunk",     seed_dt="2026-03-01")
# wm.register_org("_pipeline", stage="vectorize", seed_dt="2026-03-01")

# # COMMAND ----------

# wm.status()