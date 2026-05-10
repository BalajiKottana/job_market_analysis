"""
mcp_linkedin/server.py
──────────────────────
A small Model Context Protocol server that lets an LLM agent (Claude in
Cowork, Claude Code, or any MCP-aware client) manage your LinkedIn draft
queue and post approved drafts to LinkedIn.

Why this exists:
  build_agents.py writes draft posts to the Delta table linkedin_drafts.
  We want a controlled way to review, edit, and publish those drafts
  without giving the agent unsupervised access to LinkedIn's API.

Tools exposed:
  list_pending_drafts(limit)              → list pending drafts
  get_draft(draft_id)                     → fetch full draft for review
  approve_and_post(draft_id, edited_text) → push to LinkedIn, mark posted
  reject_draft(draft_id, note)            → mark rejected with a note

Auth:
  - Databricks  : DATABRICKS_HOST + DATABRICKS_TOKEN + DATABRICKS_HTTP_PATH
                  (a SQL warehouse path — the MCP uses databricks-sql
                  to read/update the linkedin_drafts table).
  - LinkedIn    : LINKEDIN_ACCESS_TOKEN + LINKEDIN_AUTHOR_URN
                  (member URN, e.g. "urn:li:person:abcd1234").
                  Generate via LinkedIn Developer App + OAuth 2.0
                  authorization code flow with the w_member_social scope.

Run:
  python -m mcp_linkedin.server          # stdio transport for local clients

Wiring into Cowork / Claude Code:
  Add to your client's MCP config:
    {
      "mcpServers": {
        "linkedin-drafts": {
          "command": "python",
          "args": ["-m", "mcp_linkedin.server"],
          "env": {
            "DATABRICKS_HOST":      "...",
            "DATABRICKS_TOKEN":     "...",
            "DATABRICKS_HTTP_PATH": "/sql/1.0/warehouses/...",
            "LINKEDIN_ACCESS_TOKEN": "...",
            "LINKEDIN_AUTHOR_URN":   "urn:li:person:..."
          }
        }
      }
    }
"""

from __future__ import annotations

import os
import json
from typing import Optional

import requests
from databricks import sql as dbsql
from mcp.server.fastmcp import FastMCP


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Catalog / schema kept here (not imported from properties.py) so this server
# can run outside the Databricks workspace, with its own Python env.
CATALOG     = os.environ.get("JOB_CATALOG", "bootcamp_students")
SCHEMA      = os.environ.get("JOB_SCHEMA",  "zachy_balaji_kottana05")
DRAFTS_TBL  = f"{CATALOG}.{SCHEMA}.linkedin_drafts"

DBX_HOST       = os.environ["DATABRICKS_HOST"].rstrip("/").replace("https://", "")
DBX_TOKEN      = os.environ["DATABRICKS_TOKEN"]
DBX_HTTP_PATH  = os.environ["DATABRICKS_HTTP_PATH"]

LI_TOKEN       = os.environ.get("LINKEDIN_ACCESS_TOKEN")
LI_AUTHOR_URN  = os.environ.get("LINKEDIN_AUTHOR_URN")

LI_POSTS_URL   = "https://api.linkedin.com/v2/ugcPosts"


# ─────────────────────────────────────────────────────────────────────────────
# DATABRICKS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _connect():
    return dbsql.connect(
        server_hostname = DBX_HOST,
        http_path       = DBX_HTTP_PATH,
        access_token    = DBX_TOKEN,
    )


def _query(sql: str, params: tuple = ()) -> list[dict]:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _execute(sql: str, params: tuple = ()) -> int:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount or 0


# ─────────────────────────────────────────────────────────────────────────────
# LINKEDIN POSTING
#
# Uses the v2 ugcPosts endpoint. Member-level posts only — for company-page
# posting, swap the author URN to "urn:li:organization:<id>" and ensure the
# token has the w_organization_social scope.
# ─────────────────────────────────────────────────────────────────────────────

def _post_to_linkedin(text: str) -> dict:
    if not (LI_TOKEN and LI_AUTHOR_URN):
        raise RuntimeError(
            "LINKEDIN_ACCESS_TOKEN and LINKEDIN_AUTHOR_URN must be set "
            "to publish posts. See mcp_linkedin/README.md."
        )

    headers = {
        "Authorization":              f"Bearer {LI_TOKEN}",
        "Content-Type":                "application/json",
        "X-Restli-Protocol-Version":   "2.0.0",
    }
    body = {
        "author":         LI_AUTHOR_URN,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary":    {"text": text},
                "shareMediaCategory": "NONE",
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC",
        },
    }
    resp = requests.post(LI_POSTS_URL, headers=headers,
                         data=json.dumps(body), timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"LinkedIn API {resp.status_code}: {resp.text}")
    # The post URN is in the x-restli-id header; we synthesize a permalink.
    urn = resp.headers.get("x-restli-id") or resp.json().get("id", "")
    permalink = f"https://www.linkedin.com/feed/update/{urn}/" if urn else ""
    return {"urn": urn, "url": permalink, "raw": resp.json() if resp.content else {}}


# ─────────────────────────────────────────────────────────────────────────────
# MCP SERVER
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP("linkedin-drafts")


@mcp.tool()
def list_pending_drafts(limit: int = 10) -> list[dict]:
    """List drafts awaiting review. Returns id, generated_at, intent, and the
    first 280 characters of the post text for previewing."""
    rows = _query(
        f"""
        SELECT draft_id, generated_at, intent, source_query,
               substring(post_text, 1, 280) AS preview
        FROM {DRAFTS_TBL}
        WHERE status = 'pending'
        ORDER BY generated_at
        LIMIT {int(limit)}
        """,
    )
    return rows


@mcp.tool()
def get_draft(draft_id: str) -> dict:
    """Return the full draft row, including the post_text and the underlying
    analysis_md the agent produced. Use this before approve_and_post."""
    rows = _query(
        f"SELECT * FROM {DRAFTS_TBL} WHERE draft_id = ?",
        (draft_id,),
    )
    if not rows:
        raise ValueError(f"draft_id not found: {draft_id}")
    return rows[0]


@mcp.tool()
def approve_and_post(draft_id: str,
                     edited_text: Optional[str] = None) -> dict:
    """
    Push a draft to LinkedIn and mark it posted. If edited_text is provided
    that text is what gets published (and stored on the row); otherwise the
    original post_text is used.

    Refuses to post drafts that are already 'posted' or 'rejected'.
    """
    draft = get_draft(draft_id)
    if draft["status"] in ("posted", "rejected"):
        raise RuntimeError(
            f"draft {draft_id} is in status '{draft['status']}' — "
            f"cannot re-post"
        )

    text = (edited_text or draft["post_text"]).strip()
    if not text:
        raise ValueError("post text is empty")

    result = _post_to_linkedin(text)

    _execute(
        f"""
        UPDATE {DRAFTS_TBL}
        SET status      = 'posted',
            posted_at   = current_timestamp(),
            posted_url  = ?,
            posted_by   = 'mcp',
            edited_text = COALESCE(?, edited_text)
        WHERE draft_id = ?
        """,
        (result["url"], edited_text, draft_id),
    )
    return {"draft_id": draft_id, **result}


@mcp.tool()
def reject_draft(draft_id: str, note: str = "") -> dict:
    """Mark a draft rejected with an optional note. Does not call LinkedIn."""
    _execute(
        f"""
        UPDATE {DRAFTS_TBL}
        SET status         = 'rejected',
            rejection_note = ?
        WHERE draft_id = ? AND status = 'pending'
        """,
        (note, draft_id),
    )
    return {"draft_id": draft_id, "status": "rejected", "note": note}


if __name__ == "__main__":
    mcp.run()
