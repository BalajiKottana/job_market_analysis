# Databricks notebook source
# ══════════════════════════════════════════════════════════════════════════════
# 07_agents.py — LangGraph multi-agent job market analysis system
#
# Architecture: message-passing graph with tool-calling LLM
#
# Key design decisions driven by the imports:
#
#   add_messages      — state["messages"] is a list of AnyMessage that grows
#                       across the graph. Each node appends its output as a
#                       new AIMessage rather than writing to a named field.
#                       The reducer merges lists automatically on parallel branches.
#
#   ToolNode          — prebuilt node that reads tool_calls from the last
#                       AIMessage, dispatches to the matching @tool function,
#                       and appends ToolMessage results back to state["messages"].
#
#   RunnableLambda    — wraps each agent's retrieve+prompt logic as a LangChain
#                       Runnable so it composes cleanly with the LLM chain and
#                       can be passed to RunnableConfig for tracing/callbacks.
#
#   RunnableConfig    — passed through graph.invoke() to carry callbacks,
#                       tags, and metadata (e.g. MLflow tracing) without
#                       changing any node signatures.
#
#   AIMessage         — explicit type used when constructing messages in nodes
#                       so the type checker and LangGraph reducers know the role.
#
#   AnyMessage        — union type (HumanMessage | AIMessage | ToolMessage | ...)
#                       used in the state annotation so add_messages accepts any role.
#
# Graph flow:
#   START → orchestrator_llm → tools_node (parallel tool calls) → synthesizer
#         → [if linkedin requested] → linkedin → END
#         → [else]                             → END
#
#   The orchestrator LLM decides which tools to call based on the user query.
#   ToolNode executes them concurrently. Each tool returns a ToolMessage.
#   The synthesizer reads all ToolMessages from state and merges them.
#   LinkedIn post is only generated when explicitly requested.
#
# Install (run once, then restart Python):
#   %pip install --upgrade databricks-langchain langchain langgraph
#   dbutils.library.restartPython()
# ══════════════════════════════════════════════════════════════════════════════

# COMMAND ----------

import json
import re
from typing import Annotated, Optional
from datetime import date, timedelta

# ── LangChain core — message types and runnables ─────────────
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool

# ── LangGraph — graph, message reducer, prebuilt tool node ───
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ── Databricks LLM ───────────────────────────────────────────
from databricks_langchain import ChatDatabricks

from retriever import retrieve, mmr_rerank, format_context

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

TREND_DAYS    = 30
DEFAULT_TOP_K = 15

# 70B — orchestration, synthesis, LinkedIn, QA reasoning
llm = ChatDatabricks(
    endpoint    = "databricks-meta-llama-3-3-70b-instruct",
    temperature = 0,
    max_tokens  = 2048,
)

# 8B — faster tool execution (trend, geo, role analysis prompts)
llm_fast = ChatDatabricks(
    endpoint    = "databricks-meta-llama-3-1-8b-instruct",
    temperature = 0,
    max_tokens  = 1024,
)


# ─────────────────────────────────────────────────────────────
# STATE
#
# add_messages is a reducer — when parallel branches both append
# to state["messages"], LangGraph merges the two lists rather
# than one overwriting the other. This is what makes parallel
# tool execution work correctly.
#
# AnyMessage is the union of HumanMessage | AIMessage | ToolMessage
# so the annotation accepts any role the graph produces.
# ─────────────────────────────────────────────────────────────

class AgentState(dict):
    """
    Graph state. All nodes read and write to this dict.

    messages:  Annotated[list[AnyMessage], add_messages]
        Growing conversation thread. add_messages reducer handles
        merging when parallel branches both write simultaneously.

    filters:   dict extracted by the orchestrator from the user query.
    intent:    str — "trend" | "geo" | "role" | "general"
    query:     str — original user question
    generate_linkedin: bool — whether to generate a LinkedIn post
    """
    messages:           Annotated[list[AnyMessage], add_messages]
    filters:            dict
    intent:             str
    query:              str
    final_answer:       Optional[str]
    linkedin_post:      Optional[str]
    generate_linkedin:  bool


# ─────────────────────────────────────────────────────────────
# TOOLS
#
# Each @tool function is a named capability the orchestrator LLM
# can call by name. ToolNode reads tool_calls from the last
# AIMessage, dispatches here, and wraps the return value in a
# ToolMessage that goes back into state["messages"].
#
# The tools themselves use RunnableLambda internally so they can
# be composed with LangChain chains and traced by MLflow/LangSmith.
# ─────────────────────────────────────────────────────────────

@tool
def trend_analysis_tool(query: str, filters_json: str = "{}") -> str:
    """
    Analyse skill and technology trends from job postings.
    Use for: in-demand skills, emerging vs legacy tools, experience requirements,
    AWS vs GCP stack, month-over-month changes (Amazon only).
    """
    filters = json.loads(filters_json)
    filters["section_type"] = "qualifications_required"

    # RunnableLambda wraps the retrieve call so it participates in
    # LangChain's tracing and callback system (MLflow, LangSmith)
    retrieve_fn = RunnableLambda(lambda q: retrieve(
        query      = q,
        filters    = filters,
        top_k      = 25,
        query_type = "hybrid",  # tech names need BM25
    ))

    chunks  = retrieve_fn.invoke(
        f"required skills tools technologies experience years {query}"
    )
    chunks  = mmr_rerank(chunks, query, lambda_mult=0.5, top_k=15)
    chunks.sort(key=lambda x: x["trend_dt"])
    context = format_context(chunks)

    PROMPT = f"""
You are a job market analyst. Using ONLY the job posting excerpts below, identify:

1. Top 5 in-demand skills / tools appearing most frequently
2. Emerging technologies (GenAI, Agentic AI, LLMs, vector DBs, dbt, Iceberg)
   vs traditional (Hadoop, Hive, legacy ETL tools)
3. AWS stack (Amazon) vs GCP stack (Google) where present
4. Experience requirements: note specific year ranges
5. Any temporal trends if postings span multiple months

DATA QUALITY NOTES:
- Amazon seniority "unavailable" = no level signal — do not infer
- Google seniority explicitly stated as Early/Mid/Advanced
- >>> marks the focal sentence; surrounding lines are context

Postings (sorted by date):
{context}
"""
    resp = llm_fast.invoke([HumanMessage(content=PROMPT)])
    return resp.content


@tool
def geo_analysis_tool(query: str, filters_json: str = "{}") -> str:
    """
    Analyse geographic hiring patterns from job postings.
    Use for: hiring locations, city/country concentration, regional skill differences,
    salary visibility by region, remote/hybrid/on-site signals.
    """
    filters = json.loads(filters_json)

    retrieve_fn = RunnableLambda(lambda q: retrieve(
        query      = q,
        filters    = filters,
        top_k      = 15,
        query_type = "ANN",  # location semantics — no BM25 benefit
    ))

    chunks  = retrieve_fn.invoke(
        f"location office remote hybrid work arrangement {query}"
    )
    context = format_context(chunks)

    PROMPT = f"""
You are a geographic job market analyst. Using the postings below, identify:

1. Locations with highest hiring volume — group by company if multiple are present
2. Role or skill differences by geography
3. Salary data visibility — note which regions include salary and which omit it
4. Remote/hybrid/on-site signals where mentioned

IMPORTANT: Only discuss companies and locations that appear in the retrieved postings below.
Do NOT introduce facts about companies not present in the data.

Postings:
{context}
"""
    resp = llm_fast.invoke([HumanMessage(content=PROMPT)])
    return resp.content


@tool
def role_analysis_tool(query: str, filters_json: str = "{}") -> str:
    """
    Analyse role structure, seniority levels, and responsibilities from job postings.
    Use for: DE II vs III scope, leadership signals, team domain patterns,
    AI/ML integration in data roles, salary ranges (US only).
    """
    filters = json.loads(filters_json)
    filters["section_type"] = "responsibilities"

    retrieve_fn = RunnableLambda(lambda q: retrieve(
        query      = q,
        filters    = filters,
        top_k      = 15,
        query_type = "hybrid",  # level markers (I/II/III) need BM25
    ))

    chunks  = retrieve_fn.invoke(
        f"responsibilities ownership scope leadership {query}"
    )
    context = format_context(chunks)

    PROMPT = f"""
You are a job role analyst. Analyse role structures from the postings below.
Only discuss companies that appear in the data.

NAMING CONVENTIONS:
- Google: Software Engineer II / III / Senior / Staff + seniority stated
- Amazon: Data Engineer / Data Engineer I / II / III + team suffix
  (I = Junior ~1yr, II = Mid ~3-5yr, III = Senior ~5yr+)
  Plain "Data Engineer" with no suffix = level unknown

ANALYSE:
1. Responsibility scope per level (DE II vs DE III ownership)
2. Leadership signals: mentoring, architecture, cross-functional collaboration
3. AI/ML integration: GenAI assistant, text-to-SQL, agentic pipelines
4. Team domain patterns: which teams hire at which levels
5. Salary ranges where present (US roles only)

Postings:
{context}
"""
    resp = llm_fast.invoke([HumanMessage(content=PROMPT)])
    return resp.content


@tool
def qa_check_tool(trend: str, geo: str, role: str) -> str:
    """
    Cross-check analyses against known data limitations.
    Always call this after collecting trend, geo, or role analyses.
    Returns confidence scores and flags unsupported claims.
    """
    PROMPT = f"""
You are a data quality reviewer. Cross-check these analyses against known limitations.

KNOWN DATA LIMITATIONS:
1. Amazon seniority: only I/II/III suffixes give level. Plain "Data Engineer" = unknown.
   Flag any seniority claim without confirmed suffix as speculation.
2. Salary: only US postings include salary. Flag non-US salary comparisons.
3. Google data: primarily software engineers, NOT data engineers.
   Flag direct Google vs Amazon data engineering skill comparisons.
4. Date range: Amazon spans Sep 2025-Mar 2026 (trends possible).
   Google = Mar 2026 only (single snapshot, no trend analysis).
5. Data Center Operations = physical infrastructure, NOT data engineering.
6. Internship roles should not mix with regular hire analysis unless requested.

FOR EACH ISSUE: quote the specific claim, explain why it is unsupported.
END WITH: Confidence score per section — Low / Medium / High

Trend analysis: {trend or "Not available"}
Geographic analysis: {geo or "Not available"}
Role analysis: {role or "Not available"}
"""
    resp = llm.invoke([HumanMessage(content=PROMPT)])
    return resp.content


# Register all tools — ToolNode discovers them from this list
TOOLS = [trend_analysis_tool, geo_analysis_tool, role_analysis_tool, qa_check_tool]


# ─────────────────────────────────────────────────────────────
# ORCHESTRATOR NODE
#
# The LLM is bound to the tool list so it can produce tool_calls
# in its AIMessage. It decides which tools to call and with what
# arguments based on the user query.
#
# RunnableConfig is threaded through so callers can attach
# callbacks (MLflow autologging, LangSmith tracing, etc.) to
# graph.invoke() without changing this node.
# ─────────────────────────────────────────────────────────────

ORCH_SYSTEM = """
You are a job market analysis orchestrator. You have four tools:
- trend_analysis_tool  : skill trends, technology demand, emerging vs legacy tools
- geo_analysis_tool    : hiring locations, regional differences, salary visibility
- role_analysis_tool   : role structure, seniority levels, responsibilities
- qa_check_tool        : cross-check analyses for data quality issues (always call last)

RULES:
1. Always call qa_check_tool after collecting analyses — pass the results as arguments.
2. For a general query, call trend + geo + role + qa in parallel where possible.
3. For a focused query, call only the relevant analysis tool(s) + qa.
4. Never set seniority filter for Amazon unless the user explicitly says "Data Engineer II/III".
5. Always set exclude_interns=true in filters unless user asks about internships.
6. Pass filters as a JSON string in the filters_json argument.
7. Always set org_key when the user mentions a specific company.

Available filter keys and their EXACT allowed values:
  org_key:    "google", "amazon", "youtube"  (always lowercase)
  job_family: "Data Engineering", "Software Engineering", "Data Science",
              "AI/GenAI", "Solutions Architecture", "Technical Program/Product",
              "Data Center Operations", "Other"  (title case, use category not job title)
  seniority:  "Entry", "Mid", "Senior", "Principal", "Unspecified"
              (Amazon: DE I="Entry", DE II="Mid", DE III="Senior")
              (pass a SINGLE value, not comma-separated)
  country:    country name string
  section_type: "qualifications_required", "responsibilities" (set by tools, not you)
  trend_dt:   year string like "2025" or "2026"
  exclude_interns: true/false
"""

# Bind tools to LLM so it knows what it can call
orchestrator_llm = llm.bind_tools(TOOLS)

def orchestrator_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Sends the user query to the tool-calling LLM.
    The LLM responds with an AIMessage that contains tool_calls.
    ToolNode reads those tool_calls in the next step.
    RunnableConfig carries tracing callbacks from the caller.
    """
    messages = [
        HumanMessage(content=ORCH_SYSTEM),
        HumanMessage(content=state["query"]),
    ]
    # Any existing messages (e.g. from a retry) are included
    if state.get("messages"):
        messages = state["messages"] + [HumanMessage(content=state["query"])]

    response = orchestrator_llm.invoke(messages, config=config)

    # Extract intent from tool_calls for routing metadata
    tool_names = [tc["name"] for tc in (response.tool_calls or [])]
    intent = "general"
    if tool_names == ["trend_analysis_tool"]:
        intent = "trend"
    elif tool_names == ["geo_analysis_tool"]:
        intent = "geo"
    elif tool_names == ["role_analysis_tool"]:
        intent = "role"

    return {
        "messages": [response],   # add_messages appends this AIMessage
        "intent":   intent,
    }


# ─────────────────────────────────────────────────────────────
# TOOL NODE
#
# ToolNode is a prebuilt LangGraph node. It:
#   1. Reads tool_calls from the last AIMessage in state["messages"]
#   2. Calls the matching @tool function with the provided arguments
#   3. Wraps each return value in a ToolMessage
#   4. Appends all ToolMessages to state["messages"] via add_messages
#
# Parallel tool calls (when the LLM requests multiple tools at once)
# are handled automatically — ToolNode runs them concurrently.
# ─────────────────────────────────────────────────────────────

tool_node = ToolNode(TOOLS)


# ─────────────────────────────────────────────────────────────
# ROUTING FUNCTIONS
#
# should_use_tools: After the orchestrator runs, check whether
# the AIMessage contains tool_calls.
#
# should_generate_linkedin: After the synthesizer runs, check
# whether the user requested a LinkedIn post.
# ─────────────────────────────────────────────────────────────

def should_use_tools(state: AgentState) -> str:
    """
    Reads the last message. If it has tool_calls, execute them.
    Otherwise go straight to synthesis.
    """
    last_msg = state["messages"][-1] if state.get("messages") else None
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return "synthesizer"


def should_generate_linkedin(state: AgentState) -> str:
    """
    After synthesis, check if a LinkedIn post was requested.
    Routes to linkedin node or directly to END.
    """
    if state.get("generate_linkedin", False):
        return "linkedin"
    return "end"


# ─────────────────────────────────────────────────────────────
# SYNTHESIZER NODE
#
# After tools run, state["messages"] contains ToolMessages with
# each analysis result. This node collects them and asks the LLM
# to merge into a structured report.
# ─────────────────────────────────────────────────────────────

SYNTH_PROMPT = """
You have received analyses from specialist tools. Combine them into a clear
job market report using these headers. Be factual and data-backed.
Where data is limited (Amazon seniority, missing salary, Google single snapshot),
state the limitation explicitly rather than speculating.

## Executive summary
2-3 sentences. Key finding only.

## Skill and technology trends
(from trend_analysis_tool output, or "Not analysed" if not called)

## Geographic hiring patterns
(from geo_analysis_tool output, or "Not analysed" if not called)

## Role and seniority landscape
(from role_analysis_tool output, or "Not analysed" if not called)

## Data quality and confidence
(from qa_check_tool output — include all flagged issues and confidence scores)
"""

def synthesizer_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Collects all ToolMessage content from state["messages"] and
    builds the final structured report.
    """
    # Pull tool results from message history
    tool_results = {
        msg.name: msg.content
        for msg in state["messages"]
        if isinstance(msg, ToolMessage)
    }

    context_block = "\n\n".join(
        f"[{name}]\n{content}"
        for name, content in tool_results.items()
    )

    prompt = f"{SYNTH_PROMPT}\n\nTool outputs:\n{context_block}"
    resp   = llm.invoke([HumanMessage(content=prompt)], config=config)

    return {
        "messages":     [AIMessage(content=resp.content, name="synthesizer")],
        "final_answer": resp.content,
    }


# ─────────────────────────────────────────────────────────────
# LINKEDIN NODE
#
# Only reached when generate_linkedin=True in state.
# Converts the synthesized report into a LinkedIn post.
# ─────────────────────────────────────────────────────────────

LINKEDIN_PROMPT = """
Write a LinkedIn post from this job market analysis.

STRICT RULES:
- 150-250 words maximum
- Start with a bold hook (stat or surprising observation) — never "I" or "We"
- Short punchy paragraphs (2-3 lines each)
- Include 3-5 specific data points (skill names, counts, locations)
- End with a thought-provoking question
- 5-8 relevant hashtags on the final line
- No buzzwords: no "exciting", "thrilled", "passionate", "leverage", "synergy"
- If comparing Google vs Amazon: only state what is directly data-backed
- If Amazon seniority was inferred (not stated): say "based on title patterns"
- Do NOT fabricate statistics not present in the analysis

Today: {today}
Analysis window: {window}

Analysis:
{analysis}
"""

def linkedin_node(state: AgentState, config: RunnableConfig) -> dict:
    today  = date.today()
    window = f"{today - timedelta(days=TREND_DAYS)} to {today}"

    resp = llm.invoke(
        [HumanMessage(content=LINKEDIN_PROMPT.format(
            today    = today.strftime("%B %d, %Y"),
            window   = window,
            analysis = state.get("final_answer", ""),
        ))],
        config=config,
    )
    return {
        "messages":     [AIMessage(content=resp.content, name="linkedin")],
        "linkedin_post": resp.content,
    }


# ─────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
#
# Graph flow:
#   START → orchestrator → [tools] → synthesizer
#         → [if generate_linkedin] → linkedin → END
#         → [else]                            → END
# ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("tools",        tool_node)           # ToolNode instance
    builder.add_node("synthesizer",  synthesizer_node)
    builder.add_node("linkedin",     linkedin_node)

    builder.set_entry_point("orchestrator")

    # After orchestrator: check if tool calls were requested
    builder.add_conditional_edges(
        "orchestrator",
        should_use_tools,
        {
            "tools":       "tools",       # LLM wants to call tools → execute them
            "synthesizer": "synthesizer", # no tool calls → synthesize directly
        }
    )

    # After tools execute: always synthesize
    builder.add_edge("tools", "synthesizer")

    # After synthesis: conditionally generate LinkedIn post
    builder.add_conditional_edges(
        "synthesizer",
        should_generate_linkedin,
        {
            "linkedin": "linkedin",       # user requested a LinkedIn post
            "end":      END,              # skip LinkedIn, go straight to END
        }
    )

    builder.add_edge("linkedin", END)

    return builder.compile()


graph = build_graph()


# ─────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────

def run_query(
    query:              str,
    config:             RunnableConfig = None,
    generate_linkedin:  bool = False,
) -> dict:
    """
    Run a job market query through the full agent graph.

    Args:
        query:   Natural language question. Examples:
                   "What Python and Spark skills is Amazon hiring for?"
                   "Where are most data engineering jobs located at Google?"
                   "What is the difference between DE II and DE III at Amazon?"
                   "Give me a full data engineering market overview."

        config:  Optional RunnableConfig. Use to attach callbacks such as
                 MLflow autologging or LangSmith tracing without changing
                 node signatures. Example:
                   from langchain_core.runnables import RunnableConfig
                   cfg = RunnableConfig(tags=["job-market"], metadata={"run": "daily"})
                   result = run_query("...", config=cfg)

        generate_linkedin:  Whether to generate a LinkedIn post (default: False).
                            Also auto-enabled when query contains "linkedin".
                            Examples:
                              run_query("...")                          # no post
                              run_query("...", generate_linkedin=True)  # with post
                              run_query("... and write a linkedin post") # auto-detected

    Returns:
        dict with keys:
            final_answer  — structured markdown report
            linkedin_post — LinkedIn content (None if not requested)
            messages      — full message history (AnyMessage list)
            intent        — detected intent string
    """
    # Auto-detect LinkedIn request from query text
    if not generate_linkedin and "linkedin" in query.lower():
        generate_linkedin = True

    initial_state = {
        "messages":          [],
        "query":             query,
        "intent":            "",
        "filters":           {},
        "final_answer":      None,
        "linkedin_post":     None,
        "generate_linkedin": generate_linkedin,
    }
    return graph.invoke(initial_state, config=config or {})