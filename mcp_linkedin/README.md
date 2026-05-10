# LinkedIn Drafts MCP Server

A Model Context Protocol server that exposes the `linkedin_drafts` Delta table
in your Databricks workspace as a set of tools your AI client can call.

This is the bridge between **agent generation** (the `linkedin_node` in
`build_agents.py`, which writes pending drafts to Delta) and **publication**
(LinkedIn's UGC Posts API). A human approves what goes live.

## Tools

| Tool | What it does |
|---|---|
| `list_pending_drafts(limit)` | Returns drafts awaiting review (id, intent, preview). |
| `get_draft(draft_id)` | Full draft row including the analysis the post was based on. |
| `approve_and_post(draft_id, edited_text=None)` | Publishes to LinkedIn, marks the row `posted`. |
| `reject_draft(draft_id, note)` | Marks the row `rejected`. |

## One-time setup

### 1. Install deps

```bash
pip install -r mcp_linkedin/requirements.txt
```

### 2. Databricks SQL warehouse access

You need a SQL warehouse the MCP can connect to. From the Databricks UI:

* SQL → SQL Warehouses → pick a warehouse → **Connection details**
* Copy the **Server hostname** and **HTTP path**
* Settings → Developer → Access tokens → generate a personal access token

Set:

```bash
export DATABRICKS_HOST="https://<your-workspace>.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/abcd1234"
```

If your catalog/schema differ from the project defaults
(`bootcamp_students.zachy_balaji_kottana05`), also set:

```bash
export JOB_CATALOG="..."
export JOB_SCHEMA="..."
```

### 3. LinkedIn Developer App + OAuth

1. Visit https://www.linkedin.com/developers/apps and create an app.
2. Under **Products**, request **Share on LinkedIn** (gives the
   `w_member_social` scope) and, if you'll post on a company page,
   **Marketing Developer Platform** (`w_organization_social`).
3. Under **Auth**, add an OAuth 2.0 redirect URL (any valid HTTPS URL works
   for local testing — you'll just paste back the `code` query param).
4. Run the authorization-code flow to get an access token. The simplest path
   is the small one-page Python script at the bottom of this README.

Set:

```bash
export LINKEDIN_ACCESS_TOKEN="..."
# Member URN — fetch once with:
#   curl -H "Authorization: Bearer $LINKEDIN_ACCESS_TOKEN" \
#        https://api.linkedin.com/v2/userinfo
# then format as "urn:li:person:<sub>"
export LINKEDIN_AUTHOR_URN="urn:li:person:abcd1234"
```

LinkedIn's member tokens are short-lived (60 days). Save the refresh token
from the OAuth response and rotate before expiry.

### 4. Wire to your MCP client

For Claude Code or Cowork mode, add to your MCP config:

```json
{
  "mcpServers": {
    "linkedin-drafts": {
      "command": "python",
      "args": ["-m", "mcp_linkedin.server"],
      "env": {
        "DATABRICKS_HOST":      "https://<workspace>.cloud.databricks.com",
        "DATABRICKS_TOKEN":     "dapi...",
        "DATABRICKS_HTTP_PATH": "/sql/1.0/warehouses/...",
        "LINKEDIN_ACCESS_TOKEN": "...",
        "LINKEDIN_AUTHOR_URN":   "urn:li:person:..."
      }
    }
  }
}
```

Restart your client. You should now see `linkedin-drafts` in the available
tool list.

## Typical flow

1. In Databricks, run an agent query that produces a LinkedIn draft:
   ```python
   from build_agents import run_query
   run_query("Where are most data engineering jobs and write a linkedin post")
   ```
   The post lands in `linkedin_drafts` with `status='pending'`.

2. From your MCP client, ask: *"List my pending LinkedIn drafts."*
   The client calls `list_pending_drafts`.

3. Review, optionally edit:
   *"Post draft `abc123` but change the opening line to ..."*
   The client calls `approve_and_post(draft_id="abc123", edited_text="...")`.

4. The post goes live; the row flips to `status='posted'` with the URL.

## OAuth bootstrap helper

If you don't have an access token yet, save this as `get_token.py` and run
it once:

```python
# get_token.py
import urllib.parse, webbrowser, http.server, threading, requests, os, json

CLIENT_ID     = "<your client id>"
CLIENT_SECRET = "<your client secret>"
REDIRECT_URI  = "http://localhost:8765/callback"
SCOPES        = "openid profile w_member_social"

auth_url = "https://www.linkedin.com/oauth/v2/authorization?" + urllib.parse.urlencode({
    "response_type": "code",
    "client_id":     CLIENT_ID,
    "redirect_uri":  REDIRECT_URI,
    "scope":         SCOPES,
    "state":         "x",
})
print("Open this URL in a browser:\n", auth_url)
webbrowser.open(auth_url)

code = input("Paste the `code` query param from the redirected URL: ").strip()
tok = requests.post(
    "https://www.linkedin.com/oauth/v2/accessToken",
    data={
        "grant_type":    "authorization_code",
        "code":          code,
        "redirect_uri":  REDIRECT_URI,
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    },
).json()
print(json.dumps(tok, indent=2))
```

The response contains `access_token` (use as `LINKEDIN_ACCESS_TOKEN`) and
`refresh_token` (store securely; use to get fresh tokens before the 60-day
expiry).

## Safety

* The MCP refuses to re-post drafts already `posted` or `rejected`.
* Edits go to a separate `edited_text` column; the original `post_text`
  is preserved for audit.
* `posted_url` is captured so you can find the live post later.
* No token is logged.
