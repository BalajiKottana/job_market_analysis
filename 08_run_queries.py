# Databricks notebook source
# ══════════════════════════════════════════════════════════════════════════════
# 08_run_queries.py
#
# How to use agents.py to get job market analysis results.
#
# Prerequisites:
#   1. Run bootstrap_watermark.py once (creates tables)
#   2. Run scrape_google_careers.py and scrape_amazon_careers.py
#   3. Run clean_loaded_dat.py
#   4. Run chunker.py
#   5. Run embedding.py  (creates and syncs the VS index)
#   6. Run retriever.py and agents.py    
#   6. Install packages:
#      %pip install --upgrade databricks-langchain langchain langgraph
#      dbutils.library.restartPython()
# ══════════════════════════════════════════════════════════════════════════════

# COMMAND ----------

# ── STEP 1: Install packages (run this cell alone, then restart) ─────────────

%pip install --upgrade databricks-langchain langchain langgraph
dbutils.library.restartPython()

# COMMAND ----------

# ── STEP 2: Import run_query from agents ─────────────────────────────────────

# agents.py and retriever.py must be in the same folder as this notebook,
# or in a path on sys.path (Databricks Repos puts the repo root on sys.path).

from build_agents import run_query, graph                   # main entry points
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

# ─────────────────────────────────────────────────────────────────────────────
# BASIC USAGE — just call run_query with a natural language question
# ─────────────────────────────────────────────────────────────────────────────

# COMMAND ----------

# ── Example 1: Skill trend analysis ──────────────────────────────────────────

# Reload modules to pick up fixes in retriever.py (stale module cache)
import importlib, retriever, agents
importlib.reload(retriever)
importlib.reload(agents)
from build_agents import run_query, graph

# Patch: source_url is not synced to the VS index
if "source_url" in retriever.COLUMNS:
    retriever.COLUMNS.remove("source_url")

result = run_query(
    "What are the top Python and Spark skills Amazon is hiring for in data engineering?"
)

# The two outputs you care about most:
print("=" * 60)
print("FINAL REPORT")
print("=" * 60)
print(result["final_answer"])

print("\n" + "=" * 60)
print("LINKEDIN POST")
print("=" * 60)
print(result["linkedin_post"])

# COMMAND ----------

# ── Example 2: Geographic analysis ───────────────────────────────────────────

# Reload modules to pick up prompt fixes in agents.py
import importlib, retriever, agents
importlib.reload(retriever)
importlib.reload(agents)
from build_agents import run_query

# Patch: source_url is not synced to the VS index
if "source_url" in retriever.COLUMNS:
    retriever.COLUMNS.remove("source_url")

result = run_query(
    "Where are most Google data engineering jobs located, "
    "and how do skill requirements differ between US and India?"
)

print(result["final_answer"])

# COMMAND ----------

# ── Example 3: Role and seniority comparison ─────────────────────────────────

# # Reload modules to pick up orchestrator prompt + filter fixes
# import importlib, retriever, agents
# importlib.reload(retriever)
# importlib.reload(agents)
# from build_agents import run_query

result = run_query(
    "What is the difference in responsibilities between "
    "a Data Engineer II and Data Engineer III at Amazon?"
)

print(result["final_answer"])

# COMMAND ----------

# ── Example 4: Full market overview (all agents) ─────────────────────────────

result = run_query(
    "Write a linkedin post describing the top 3 skills in the SF Bay Area for Data Analysts"
)

print(result["final_answer"])
# print("\n--- LINKEDIN POST ---")
# print(result["linkedin_post"])

# ─────────────────────────────────────────────────────────────────────────────
# ACCESSING INDIVIDUAL TOOL OUTPUTS
# Each tool (trend, geo, role, qa) appends a ToolMessage to state["messages"].
# You can pull them out individually if you want just one section.
# ─────────────────────────────────────────────────────────────────────────────

# COMMAND ----------

result = run_query("What GenAI skills are Amazon hiring for?")

# Pull individual tool results from the message history
tool_outputs = {
    msg.name: msg.content
    for msg in result["messages"]
    if isinstance(msg, ToolMessage)
}

# Print each tool's output separately
for tool_name, content in tool_outputs.items():
    print(f"\n{'─' * 50}")
    print(f"  {tool_name}")
    print(f"{'─' * 50}")
    print(content)

# COMMAND ----------

# ── Just the trend analysis, no geo or role ───────────────────────────────────

# The orchestrator LLM decides which tools to call based on your question.
# A focused question routes to fewer tools.

result = run_query(
    "What cloud infrastructure skills (AWS, GCP, Azure) appear most "
    "in Amazon data engineering job requirements?"
)

# Check which tools the orchestrator actually called
tool_names_called = [
    msg.name
    for msg in result["messages"]
    if isinstance(msg, ToolMessage)
]
print(f"Tools called: {tool_names_called}")
print(result["final_answer"])

# ─────────────────────────────────────────────────────────────────────────────
# USING RunnableConfig — attach MLflow tracing or tags
# ─────────────────────────────────────────────────────────────────────────────

# COMMAND ----------

import mlflow
from datetime import date

# COMMAND ----------



# MLflow autologging captures every LLM call, token count, and latency
mlflow.langchain.autolog()

cfg = RunnableConfig(
    tags     = ["job-market", "daily-analysis"],
    metadata = {"run_date": str(date.today()), "triggered_by": "notebook"},
)

result = run_query(
    "What dbt and Spark skills are trending in Amazon data engineering?",
    config=cfg,
)

print(result["final_answer"])
# MLflow UI will show a trace for every LLM + tool call in this run

# ─────────────────────────────────────────────────────────────────────────────
# INSPECTING THE FULL MESSAGE HISTORY
# state["messages"] is a list of AnyMessage — every step is recorded.
# ─────────────────────────────────────────────────────────────────────────────

# COMMAND ----------

result = run_query("What are the top 5 skills for a Data Scientist at Google?")

print(f"Total messages in conversation: {len(result['messages'])}\n")

for i, msg in enumerate(result["messages"]):
    msg_type = type(msg).__name__
    name     = getattr(msg, "name", "") or ""
    preview  = str(msg.content)[:120].replace("\n", " ")
    print(f"[{i}] {msg_type:<15} {name:<20} {preview}...")

# ─────────────────────────────────────────────────────────────────────────────
# BATCH QUERIES — run multiple questions and collect results
# ─────────────────────────────────────────────────────────────────────────────

# COMMAND ----------

QUESTIONS = [
    "What Python skills does Amazon require for data engineering roles in India?",
    "Compare seniority levels and responsibilities for data engineering at Amazon US vs India.",
    "What GenAI and LLM skills are appearing in Google job requirements?",
    "Which Amazon team domains (Fintech, AWS, Advertising) hire the most data engineers?",
]

results = []
for q in QUESTIONS:
    print(f"\nRunning: {q[:60]}...")
    r = run_query(q)
    results.append({
        "query":        q,
        "answer":       r["final_answer"],
        "linkedin":     r["linkedin_post"],
        "tools_called": [m.name for m in r["messages"] if isinstance(m, ToolMessage)],
    })
    print(f"  Tools used: {results[-1]['tools_called']}")

# Print all answers
for r in results:
    print("\n" + "=" * 70)
    print(f"Q: {r['query']}")
    print("=" * 70)
    print(r["answer"])

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS TO A DELTA TABLE
# ─────────────────────────────────────────────────────────────────────────────

# COMMAND ----------

from pyspark.sql import Row
from datetime import date

catalog     = "bootcamp_students"
schema_name = "zachy_balaji_kottana05"
RESULTS_TABLE = f"{catalog}.{schema_name}.agent_query_results"

rows = [
    Row(
        run_date      = str(date.today()),
        query         = r["query"],
        intent        = r.get("intent",""),           # add if you track intent separately
        tools_called  = str(r["tools_called"]),
        final_answer  = r["answer"],
        linkedin_post = r["linkedin"],
    )
    for r in results
]

spark.createDataFrame(rows).write \
    .format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .saveAsTable(RESULTS_TABLE)

print(f"Saved {len(rows)} results to {RESULTS_TABLE}")

# COMMAND ----------

# ── Verify saved results ──────────────────────────────────────────────────────

display(spark.table(RESULTS_TABLE).orderBy("run_date").limit(10))

# COMMAND ----------

# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.jobs import JobSettings, CronSchedule
# WorkspaceClient().jobs.update(
#     job_id=294804090840260,
#     new_settings=JobSettings(schedule=CronSchedule(
#         quartz_cron_expression="0 0 8 * * ? *",
#         timezone_id="America/New_York",
#     ))
# )

# COMMAND ----------


