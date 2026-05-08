# Databricks notebook source
# ══════════════════════════════════════════════════════════════════════════════
# 05_embedder.py  — Databricks Vector Search orchestrator
#
# Databricks-native stack:
#   - Embedding:    databricks-gte-large-en  (Databricks-hosted, no API key)
#   - Vector store: Databricks Vector Search  (Delta Sync index, TRIGGERED)
#   - Index type:   DELTA_SYNC — watches job_sentence_chunks Delta table,
#                   re-embeds automatically when we trigger a sync
#
# Watermark gate:
#   upper bound = get_stage_watermark("chunk")
#   lower bound = get_stage_watermark("vectorize")
#   window = (lower, upper]
#
# What this notebook does:
#   1. Gate on watermarks — skip if nothing new
#   2. Create the VS endpoint (idempotent)         → vs_index.py
#   3. Create the DELTA_SYNC index (idempotent)     → vs_index.py
#   4. Trigger a sync + wait for completion          → vs_sync.py
#   5. Advance the vectorize watermark
# ══════════════════════════════════════════════════════════════════════════════

from pipeline_watermark import WatermarkManager
from properties import CHUNKS_TABLE, chunk_stage, vector_stage

from vs_index import create_or_use_vector_endpoint, create_or_use_vector_index
from vs_sync import trigger_sync_and_wait

from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F


# ─────────────────────────────────────────────────────────────
# WATERMARK GATE
# ─────────────────────────────────────────────────────────────

def check_for_new_chunks(chunk_wm, vectorize_wm) -> int:
    try:
        new_chunk_count = (
            spark.table(CHUNKS_TABLE)
            .filter(F.col("file_dt") >  F.lit(vectorize_wm))
            .filter(F.col("file_dt") <= F.lit(chunk_wm))
            .count()
        )
        return new_chunk_count
    except Exception as e:
        print(f"error checking for new chunks: {e}")
        return 0


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    wm = WatermarkManager(spark)
    wm.register_org("_pipeline", stage=vector_stage, seed_dt="2024-01-01")

    chunk_wm     = wm.get_stage_watermark(chunk_stage)
    vectorize_wm = wm.get_stage_watermark(vector_stage)

    if chunk_wm is None:
        print("Chunking has not run yet — nothing to vectorize")
        return None
    if vectorize_wm is not None and vectorize_wm >= chunk_wm:
        print(f"Vectorize watermark {vectorize_wm} is current ({chunk_wm}) — nothing to do")
        return None

    new_chunk_count = check_for_new_chunks(chunk_wm, vectorize_wm)
    if new_chunk_count == 0:
        wm.mark_stage_success(vector_stage, chunk_wm, 0)
        print("No new chunks — watermark advanced, nothing to embed")
        return None

    print(f"Found {new_chunk_count} new chunks in window ({vectorize_wm}, {chunk_wm}]")

    w = WorkspaceClient()

    if not create_or_use_vector_endpoint(w, wm):
        return None
    if not create_or_use_vector_index(w, wm):
        return None
    if not trigger_sync_and_wait(w, wm, chunk_wm, new_chunk_count):
        return None

    wm.status()
    return None


if __name__ == '__main__':
    main()
