# ─────────────────────────────────────────────────────────────
# vs_index.py — Vector Search endpoint & index management
#
# Contains:
#   - create_or_use_vector_endpoint  (idempotent endpoint setup)
#   - create_or_use_vector_index     (idempotent DELTA_SYNC index)
#   - _wait_for_index_ready          (readiness polling)
# ─────────────────────────────────────────────────────────────

from pipeline_watermark import WatermarkManager
from properties import CHUNKS_TABLE, VS_INDEX, EMBED_MODEL, VS_ENDPOINT, EMBED_COLUMN, vector_stage

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType
import time


# ─────────────────────────────────────────────────────────────
# STEP 1 — CREATE VS ENDPOINT  (idempotent)
#
# The endpoint is a long-running compute resource managed by Databricks.
# Create it once — all indexes in this workspace share it.
# Safe to re-run — catches "already exists" and reuses.
# ─────────────────────────────────────────────────────────────

def create_or_use_vector_endpoint(w: WorkspaceClient, wm: WatermarkManager) -> bool:
    try:
        w.vector_search_endpoints.create_endpoint(
            name          = VS_ENDPOINT,
            endpoint_type = EndpointType.STANDARD,
        ).result()
        print(f"VS endpoint created: {VS_ENDPOINT}")
        return True
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"VS endpoint already exists: {VS_ENDPOINT} — reusing")
            return True
        else:
            wm.mark_stage_failed(vector_stage, f"endpoint creation failed: {e}")
            print(f"VS endpoint creation failed: {e}")
            return False


# ─────────────────────────────────────────────────────────────
# STEP 2 — CREATE DELTA_SYNC INDEX  (idempotent)
#
# DELTA_SYNC means Databricks watches job_sentence_chunks and re-embeds
# whenever we trigger a sync. We never call the embedding model directly —
# Databricks calls databricks-gte-large-en on the sentence column for us.
#
# primary_key = chunk_id  (already unique from chunker.py)
# embedding_source_columns = [sentence]  (focal sentence, not window_text)
#
# Metadata columns (org_key, job_family, country, section_type, trend_dt,
# seniority, is_intern) are stored in the index so the retriever can
# filter them with query_filters without a separate Delta table join.
# ─────────────────────────────────────────────────────────────

def _wait_for_index_ready(w: WorkspaceClient, poll_sec: int = 30, timeout_sec: int = 600) -> bool:
    """Poll until the index reports ready=True or timeout."""
    elapsed = 0
    while elapsed < timeout_sec:
        resp = w.api_client.do("GET", f"/api/2.0/vector-search/indexes/{VS_INDEX}")
        ready = resp.get("status", {}).get("ready", False)
        msg   = resp.get("status", {}).get("message", "")
        print(f"  [{elapsed}s] index ready: {ready}  ({msg})")
        if ready:
            return True
        time.sleep(poll_sec)
        elapsed += poll_sec
    return False


def create_or_use_vector_index(w: WorkspaceClient, wm: WatermarkManager) -> bool:
    try:
        w.api_client.do(
            "POST",
            "/api/2.0/vector-search/indexes",
            body={
                "name":         VS_INDEX,
                "endpoint_name": VS_ENDPOINT,
                "primary_key":  "chunk_id",
                "index_type":   "DELTA_SYNC",
                "delta_sync_index_spec": {
                    "source_table": CHUNKS_TABLE,
                    "pipeline_type": "TRIGGERED",
                    "embedding_source_columns": [
                        {
                            "name": EMBED_COLUMN,
                            "embedding_model_endpoint_name": EMBED_MODEL,
                        }
                    ],
                    "columns_to_sync": [
                        "chunk_id", "doc_id", "sentence", "window_text",
                        "section", "section_type", "sentence_index",
                        # multi-granularity additions
                        "granularity", "parent_chunk_id", "token_count_est",
                        "org_key", "job_id", "title_clean", "job_family",
                        "domain", "team_name", "seniority", "seniority_source",
                        "is_intern", "job_location_clean", "country",
                        "trend_dt", "file_dt",
                    ]
                }
            }
        )
        print(f"VS index created: {VS_INDEX}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"VS index already exists: {VS_INDEX} — will trigger sync")
        else:
            wm.mark_stage_failed(vector_stage, f"index creation failed: {e}")
            print(f"VS index creation failed: {e}")
            return False

    # Wait for the index to be ready before allowing sync
    print(f"Waiting for index to become ready...")
    if not _wait_for_index_ready(w):
        wm.mark_stage_failed(vector_stage, "index not ready after timeout")
        print(f"VS index not ready after timeout")
        return False
    return True
