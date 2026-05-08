# ─────────────────────────────────────────────────────────────
# vs_sync.py — Vector Search sync trigger & polling
#
# Contains:
#   - trigger_sync_and_wait  (trigger TRIGGERED sync + poll until done)
#
# Polls status.detailed_state for sync progress:
#   - ONLINE_TRIGGERED_UPDATE   → sync in progress
#   - ONLINE_NO_PENDING_UPDATE  → sync complete
#   - ONLINE_PIPELINE_FAILED    → sync failed
# ─────────────────────────────────────────────────────────────

from pipeline_watermark import WatermarkManager
from properties import VS_INDEX, vector_stage

from databricks.sdk import WorkspaceClient
import time


# ─────────────────────────────────────────────────────────────
# TRIGGER SYNC + WAIT
#
# TRIGGERED pipeline_type means the index only re-embeds when we
# explicitly call sync. We call it here after every successful chunking run.
# Databricks will embed only rows that changed since the last sync.
#
# Polling every 30 seconds. Typical sync time: 2–5 min for ~10k chunks.
# ─────────────────────────────────────────────────────────────

def trigger_sync_and_wait(w: WorkspaceClient, wm: WatermarkManager,
                         chunk_wm: str, new_chunk_count: int) -> bool:
    try:
        print(f"Triggering VS index sync: {VS_INDEX}")
        w.api_client.do(
            "POST",
            f"/api/2.0/vector-search/indexes/{VS_INDEX}/sync",
        )

        poll_interval_sec = 30
        timeout_sec       = 3600   # 60 minutes max
        elapsed           = 0

        while elapsed < timeout_sec:
            time.sleep(poll_interval_sec)
            elapsed += poll_interval_sec

            status_resp = w.api_client.do(
                "GET",
                f"/api/2.0/vector-search/indexes/{VS_INDEX}",
            )
            status_info    = status_resp.get("status", {})
            detailed_state = status_info.get("detailed_state", "UNKNOWN")
            row_count      = status_info.get("indexed_row_count", 0)

            print(f"  [{elapsed}s] state: {detailed_state}, indexed rows: {row_count}")

            if detailed_state == "ONLINE_NO_PENDING_UPDATE":
                print(f"VS index sync complete: {VS_INDEX} ({row_count} rows indexed)")
                wm.mark_stage_success(vector_stage, chunk_wm, new_chunk_count)
                return True
            elif "FAILED" in detailed_state:
                print(f"VS index sync failed: {detailed_state}")
                return False

        print(f"VS index sync timed out after {timeout_sec}s")
        return False

    except Exception as e:
        wm.mark_stage_failed(vector_stage, str(e))
        print(f"VS sync failed: {e}")
        return False
