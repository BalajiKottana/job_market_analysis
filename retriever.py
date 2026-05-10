"""
06_retriever.py  — Databricks Vector Search version

Fix applied: BadRequest: Columns referenced in filters are not present in index

Root cause — three bugs in the original build_filters():

  Bug 1 — Wrong filter format:
    Old: [{"column": "org_key", "op": "EQUAL", "value": "google"}]
    The VS SDK does NOT accept this {column, op, value} format.
    The VS API was interpreting "column", "op", "value" as column names
    in the index, which is why the error said "columns referenced in
    filters are not present in index".

    Correct format is a plain dict:
      {"org_key": "google"}             # equality
      {"org_key NOT": "google"}         # not equal
      {"trend_dt >=": "2025-01-01"}     # greater than or equal
      {"trend_dt <=": "2025-12-31"}     # less than or equal

  Bug 2 — Wrong serialisation:
    Old: body["filters_json"] = str(vs_filters).replace("'", '"')
    str() uses Python repr format (True/False capitalised, single quotes).
    Even after replacing quotes the result is not valid JSON.
    Fix: pass filters dict directly to the SDK — no serialisation needed.

  Bug 3 — Wrong boolean value:
    Old: {"value": "false"}   (string)
    Fix: {"is_intern": False}  (Python bool → JSON false)
"""

#%pip install databricks-vectorsearch

import numpy as np
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient

catalog     = "bootcamp_students"
schema_name = "zachy_balaji_kottana05"
VS_INDEX    = f"{catalog}.{schema_name}.job_chunks_index"

DEFAULT_TOP_K       = 15
CANDIDATE_POOL_MULT = 4
RRF_K               = 60

# WorkspaceClient for auth + MMR embedding calls
w   = WorkspaceClient()
# VectorSearchClient for index queries — uses correct filter dict format
vsc = VectorSearchClient()
idx = vsc.get_index(index_name=VS_INDEX)


# ─────────────────────────────────────────────────────────────────────────────
# FILTER BUILDER
#
# Databricks VS filter dict format:
#   {"column":        value}   → equality   (column = value)
#   {"column NOT":    value}   → not equal  (column != value)
#   {"column >=":     value}   → >=
#   {"column <=":     value}   → <=
#   {"column LIKE":   value}   → contains token (whitespace-delimited)
#
# Multiple keys = implicit AND.
# Pass the dict directly to similarity_search(filters=...) — no JSON needed.
# ─────────────────────────────────────────────────────────────────────────────

def build_filters(filters: dict) -> dict:
    """
    Converts our standard filter dict into the Databricks VS filter format.

    Returns a single dict with all conditions.
    Multiple keys are combined as an implicit AND by the VS engine.
    Empty dict = no filters (return all results).
    """
    vs = {}

    if filters.get("org_key"):
        vs["org_key"] = filters["org_key"].lower()

    if filters.get("job_family"):
        # Normalize common LLM outputs to actual index values
        _JF_MAP = {
            "data engineer": "Data Engineering",
            "data engineering": "Data Engineering",
            "software engineer": "Software Engineering",
            "software engineering": "Software Engineering",
            "data science": "Data Science",
            "data scientist": "Data Science",
            "ai/genai": "AI/GenAI",
        }
        jf = filters["job_family"]
        vs["job_family"] = _JF_MAP.get(jf.lower(), jf.title())

    if filters.get("domain"):
        vs["domain"] = filters["domain"]

    if filters.get("section_type"):
        vs["section_type"] = filters["section_type"]

    if filters.get("country"):
        vs["country"] = filters["country"]

    if filters.get("trend_dt"):
        yr = filters["trend_dt"]
        vs["trend_dt >="] = f"{yr}-01-01"
        vs["trend_dt <="] = f"{yr}-12-31"

    # exclude_interns defaults to True — always filter unless explicitly False
    if filters.get("exclude_interns", True):
        vs["is_intern"] = False   # Python bool → JSON false — NOT the string "false"

    # granularity: "document" | "section" | "paragraph" | "sentence"
    # Default behaviour: leave unfiltered so multi-granularity recall works.
    # Callers that need a specific level pass filters["granularity"] explicitly.
    if filters.get("granularity"):
        vs["granularity"] = filters["granularity"]

    return vs


def post_filter(chunks: list, filters: dict) -> list:
    """
    Applies filters that require multi-column logic — not expressible
    in the single-column VS filter syntax.

    seniority: requires checking both seniority = X AND seniority_source != unavailable.
    Data Center Operations: excluded when job_family = Data Engineering was requested.
    """
    if filters.get("seniority"):
        requested = filters["seniority"]
        chunks = [
            c for c in chunks
            if c["seniority"] == requested
            and c["seniority_source"] != "unavailable"
        ]

    if filters.get("job_family") == "Data Engineering":
        chunks = [c for c in chunks if c["job_family"] != "Data Center Operations"]

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# CORE RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

COLUMNS = [
    "chunk_id", "sentence", "window_text",
    "doc_id",
    # multi-granularity fields — present after the chunker is re-run
    # against CHUNKING_STRATEGY="multi_granularity". Safe to include even
    # when the index doesn't have them yet (VS just drops unknown names).
    "granularity", "parent_chunk_id", "token_count_est",
    "org_key", "title_clean", "job_family", "domain", "team_name",
    "job_location_clean", "country",
    "seniority", "seniority_source",
    "section_type", "trend_dt", "is_intern",
    "file_dt",
]

def retrieve(
    query:      str,
    filters:    dict = None,
    top_k:      int  = DEFAULT_TOP_K,
    query_type: str  = "hybrid",
) -> list:
    """
    Retrieves top_k chunks from the Databricks Vector Search index.

    Uses VectorSearchClient.similarity_search() which:
    - Accepts filter dict directly (no manual JSON serialisation)
    - Handles embedding the query server-side via databricks-gte-large-en
    - Supports query_type="ANN" (dense) or "HYBRID" (dense + BM25)

    Returns list of chunk dicts with all metadata fields.
    """
    filters    = filters or {}
    vs_filters = build_filters(filters)
    pool       = top_k * CANDIDATE_POOL_MULT

    response = idx.similarity_search(
        query_text   = query,
        columns      = COLUMNS,
        filters      = vs_filters if vs_filters else None,
        num_results  = pool,
        query_type   = query_type,
    )

    # SDK returns {"result": {"data_array": [...], "row_count": N},
    #              "manifest": {"columns": [{"name": ...}, ...]}}
    manifest_cols = [
        c["name"]
        for c in response.get("manifest", {}).get("columns", [])
    ]
    rows = response.get("result", {}).get("data_array", [])

    chunks = []
    for row in rows:
        record = dict(zip(manifest_cols, row))
        chunks.append({
            "chunk_id":         record.get("chunk_id", ""),
            "sentence":         record.get("sentence", ""),
            "window_text":      record.get("window_text", ""),
            "doc_id":           record.get("doc_id", ""),
            "source_url":       record.get("source_url", ""),
            "org_key":          record.get("org_key", ""),
            "title":            record.get("title_clean", ""),
            "job_family":       record.get("job_family", ""),
            "domain":           record.get("domain", ""),
            "team_name":        record.get("team_name", ""),
            "job_location":     record.get("job_location_clean", ""),
            "country":          record.get("country", ""),
            "seniority":        record.get("seniority", "Unspecified"),
            "seniority_source": record.get("seniority_source", "unavailable"),
            "section_type":     record.get("section_type", ""),
            "trend_dt":         str(record.get("trend_dt", "")),
            "is_intern":        record.get("is_intern", False),
            "score":            record.get("score", 0.0),
        })

    chunks = post_filter(chunks, filters)
    return chunks[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# MMR RE-RANKING
# ─────────────────────────────────────────────────────────────────────────────

def _embed_texts_local(texts: list) -> np.ndarray:
    """
    Embeds texts via Databricks Foundation Model serving endpoint.
    Used only for MMR cosine math — query retrieval embedding is server-side.
    """
    response = w.api_client.do(
        "POST",
        "/serving-endpoints/databricks-gte-large-en/invocations",
        body={"input": texts},
    )
    vecs = [item["embedding"] for item in response["data"]]
    return np.array(vecs)


def mmr_rerank(
    chunks:      list,
    query:       str,
    lambda_mult: float = 0.5,
    top_k:       int   = DEFAULT_TOP_K,
) -> list:
    """
    Maximal Marginal Relevance re-ranking.
    Reduces redundancy — prevents all 15 chunks coming from the same
    boilerplate JD when many near-identical postings exist.

    lambda_mult=0 → max diversity, lambda_mult=1 → max relevance.
    """
    if len(chunks) <= top_k:
        return chunks

    texts     = [c["sentence"] for c in chunks]
    all_vecs  = _embed_texts_local(texts + [query])
    vecs      = all_vecs[:-1]
    q_vec     = all_vecs[-1]

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    selected  = []
    remaining = list(range(len(chunks)))

    while len(selected) < top_k and remaining:
        if not selected:
            scores = [cosine(vecs[i], q_vec) for i in remaining]
            best   = remaining[int(np.argmax(scores))]
        else:
            scores = [
                lambda_mult * cosine(vecs[i], q_vec)
                - (1 - lambda_mult) * max(cosine(vecs[i], vecs[j]) for j in selected)
                for i in remaining
            ]
            best = remaining[int(np.argmax(scores))]

        selected.append(best)
        remaining.remove(best)

    return [chunks[i] for i in selected]


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

def format_context(chunks: list, include_meta: bool = True) -> str:
    """
    Formats retrieved chunks into a context string for LLM prompts.
    Uses window_text (focal sentence + surrounding context) not raw sentence.
    """
    parts = []
    for c in chunks:
        if include_meta:
            seniority_note = (
                f"seniority:{c['seniority']}({c['seniority_source']})"
                if c["seniority"] != "Unspecified"
                else "seniority:not published"
            )
            header = (
                f"[{c['org_key'].upper()} | {c['job_family']} | "
                f"{c['country']} | {seniority_note} | "
                f"{c['trend_dt']} | {c['title']}]"
            )
        else:
            header = f"[{c['org_key'].upper()} | {c['trend_dt']}]"

        parts.append(f"{header}\n{c['window_text']}")

    return "\n---\n".join(parts)