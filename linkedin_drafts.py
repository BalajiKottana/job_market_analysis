# Databricks notebook source
# ══════════════════════════════════════════════════════════════════════════════
# linkedin_drafts.py
#
# Delta-backed draft queue for LinkedIn posts produced by build_agents.py.
#
# Why a queue (not direct posting):
#   - Posts that quote skill counts, salary ranges, or company comparisons can
#     embarrass when wrong. A human-in-the-loop step is cheap insurance.
#   - The queue gives us an audit trail: which agent run produced the post,
#     who approved/rejected, when it actually went live.
#   - Decouples generation (Databricks) from posting (LinkedIn API, OAuth).
#     The MCP server in mcp_linkedin/ is the only thing that touches LinkedIn.
#
# Lifecycle:
#
#     pending  ─approve→  approved ─post→  posted
#         │
#         └──reject──→ rejected
#
# Schema (Delta, in Unity Catalog):
#
#     draft_id        STRING        sha256(source_query + generated_at_ms)
#     generated_at    TIMESTAMP     when the agent produced the draft
#     source_query    STRING        the original user query
#     intent          STRING        agent intent: trend/geo/role/general
#     analysis_md     STRING        the synthesizer's structured report
#     post_text       STRING        the LinkedIn-formatted text
#     status          STRING        pending | approved | posted | rejected
#     edited_text     STRING        user-edited final text (nullable)
#     posted_at       TIMESTAMP     when LinkedIn confirmed the post
#     posted_url      STRING        LinkedIn share URN/URL
#     posted_by       STRING        whoever ran the MCP tool
#     rejection_note  STRING        why it was rejected
#
# Idempotency: draft_id is deterministic from (source_query, generated_at).
# Calling save_draft twice for the same agent run will UPDATE rather than
# duplicate.
# ══════════════════════════════════════════════════════════════════════════════

import hashlib
from datetime import datetime
from typing import Optional

from properties import LINKEDIN_DRAFTS_TABLE


# ─────────────────────────────────────────────────────────────────────────────
# TABLE BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────

def ensure_drafts_table(spark) -> None:
    """Create the drafts table if it doesn't exist. Safe to call repeatedly."""
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {LINKEDIN_DRAFTS_TABLE} (
            draft_id        STRING    NOT NULL,
            generated_at    TIMESTAMP NOT NULL,
            source_query    STRING,
            intent          STRING,
            analysis_md     STRING,
            post_text       STRING    NOT NULL,
            status          STRING    NOT NULL,
            edited_text     STRING,
            posted_at       TIMESTAMP,
            posted_url      STRING,
            posted_by       STRING,
            rejection_note  STRING
        ) USING DELTA
        TBLPROPERTIES (
            'delta.enableChangeDataFeed' = 'true'
        )
    """)


def _draft_id_for(source_query: str, generated_at: datetime) -> str:
    h = hashlib.sha256()
    h.update(source_query.encode("utf-8"))
    h.update(str(int(generated_at.timestamp() * 1000)).encode("utf-8"))
    return h.hexdigest()[:24]


# ─────────────────────────────────────────────────────────────────────────────
# WRITE
#
# Called from build_agents.linkedin_node after the LLM produces the post text.
# ─────────────────────────────────────────────────────────────────────────────

def save_draft(
    spark,
    source_query: str,
    intent:        str,
    analysis_md:   str,
    post_text:     str,
) -> str:
    """
    MERGE a draft into LINKEDIN_DRAFTS_TABLE. Returns draft_id.
    Idempotent: same (source_query, generated_at) → same draft_id → UPDATE.
    """
    ensure_drafts_table(spark)

    generated_at = datetime.utcnow()
    draft_id     = _draft_id_for(source_query, generated_at)

    # Use parameterized SQL to avoid quoting headaches with multiline post text.
    spark.sql(
        f"""
        MERGE INTO {LINKEDIN_DRAFTS_TABLE} AS T
        USING (SELECT
                 :did                 AS draft_id,
                 CAST(:gen AS TIMESTAMP) AS generated_at,
                 :sq                  AS source_query,
                 :intent              AS intent,
                 :amd                 AS analysis_md,
                 :pt                  AS post_text,
                 'pending'            AS status
              ) AS S
        ON T.draft_id = S.draft_id
        WHEN MATCHED THEN UPDATE SET
            post_text   = S.post_text,
            analysis_md = S.analysis_md,
            status      = CASE WHEN T.status IN ('posted','rejected')
                                THEN T.status ELSE 'pending' END
        WHEN NOT MATCHED THEN INSERT (
            draft_id, generated_at, source_query, intent,
            analysis_md, post_text, status
        ) VALUES (
            S.draft_id, S.generated_at, S.source_query, S.intent,
            S.analysis_md, S.post_text, S.status
        )
        """,
        args={
            "did":    draft_id,
            "gen":    generated_at.isoformat(),
            "sq":     source_query,
            "intent": intent or "general",
            "amd":    analysis_md or "",
            "pt":     post_text,
        },
    )
    return draft_id


# ─────────────────────────────────────────────────────────────────────────────
# READ / TRANSITION HELPERS
#
# These are convenience wrappers used by the approval UI or by the MCP server
# when it runs from inside a Databricks notebook. The MCP server itself uses
# the Databricks SQL connector for the same operations.
# ─────────────────────────────────────────────────────────────────────────────

def list_pending(spark, limit: int = 25):
    return (
        spark.table(LINKEDIN_DRAFTS_TABLE)
             .filter("status = 'pending'")
             .orderBy("generated_at")
             .limit(limit)
    )


def approve(spark, draft_id: str, edited_text: Optional[str] = None) -> None:
    if edited_text is not None:
        spark.sql(
            f"""UPDATE {LINKEDIN_DRAFTS_TABLE}
                SET status='approved', edited_text=:t
                WHERE draft_id=:d""",
            args={"t": edited_text, "d": draft_id},
        )
    else:
        spark.sql(
            f"""UPDATE {LINKEDIN_DRAFTS_TABLE}
                SET status='approved'
                WHERE draft_id=:d""",
            args={"d": draft_id},
        )


def reject(spark, draft_id: str, note: str = "") -> None:
    spark.sql(
        f"""UPDATE {LINKEDIN_DRAFTS_TABLE}
            SET status='rejected', rejection_note=:n
            WHERE draft_id=:d""",
        args={"n": note, "d": draft_id},
    )


def mark_posted(spark, draft_id: str, posted_url: str,
                posted_by: str = "mcp") -> None:
    spark.sql(
        f"""UPDATE {LINKEDIN_DRAFTS_TABLE}
            SET status='posted',
                posted_at=current_timestamp(),
                posted_url=:u,
                posted_by=:b
            WHERE draft_id=:d""",
        args={"u": posted_url, "b": posted_by, "d": draft_id},
    )
