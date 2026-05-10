# Databricks notebook source
# ══════════════════════════════════════════════════════════════════════════════
# chunking_strategies.py
#
# Multi-granularity chunking for job descriptions.
#
# Stores four chunk granularities in a single table, linked by parent_chunk_id:
#
#     document   ── LLM whole-job summary (~120 tokens)         top-down retrieval
#        │                                                       (recommendation,
#        │                                                        comparison, Q&A)
#     section    ── LLM section summary (~100 tokens)           mid-level retrieval
#        │                                                       (about/quals/resp)
#     paragraph  ── rule-based ~300-token coherent block         workhorse for
#        │                                                       reasoning context
#     sentence   ── single sentence + windowed display context   atomic precision
#                                                                (skill counting,
#                                                                 fact extraction)
#
# Why this shape:
#   - Different downstream tasks want different chunk sizes. Storing all four
#     and filtering by `granularity` at query time is cheaper than re-chunking
#     for each new use case.
#   - parent_chunk_id lets a retriever expand context upward (for grounding)
#     or descend (for evidence). The retriever can pull a paragraph for
#     reasoning, then dereference its sentences when the user asks "exactly
#     where did this come from".
#
# Schema (added/kept on CHUNKS_TABLE):
#   chunk_id          STRING   primary key
#   doc_id            STRING   foreign key → clean_openings
#   granularity       STRING   "document" | "section" | "paragraph" | "sentence"
#   parent_chunk_id   STRING   nullable; null for document granularity
#   sentence          STRING   the text that gets embedded by Vector Search
#                              (we keep this column name so the existing index
#                              works without re-creation — it just holds bigger
#                              text for non-sentence rows)
#   window_text       STRING   the text returned to the LLM for display/grounding
#   section           STRING   parsed section header (best-effort)
#   section_type      STRING   normalized: about / qualifications_required /
#                              qualifications_preferred / responsibilities / other
#   sentence_index    STRING   ordering hint within the doc (e.g. "p2_s5",
#                              "section_summary", "doc_summary")
#   token_count_est   INT      char-count / 4, helps callers budget context
#   ... plus the existing job-level columns (org_key, country, seniority, etc.)
#
# Pipeline shape (called from chunker.py main()):
#
#   1. Rule-based UDF produces sentence + paragraph chunks per doc.
#   2. SQL ai_query produces section summaries (one row per (doc, section)).
#   3. SQL ai_query produces document summaries (one row per doc).
#   4. UNION all four sets, fill parent_chunk_id, MERGE into CHUNKS_TABLE.
#
# The ai_query calls are made via Spark SQL (not from inside a UDF) so they
# benefit from Databricks model-serving batching and don't block executors.
# ══════════════════════════════════════════════════════════════════════════════

import re
import pandas as pd
from pyspark.sql import functions as F, DataFrame
from pyspark.sql.types import (
    ArrayType, StructType, StructField, StringType, IntegerType,
)

from properties import (
    CLEAN_TABLE, CHUNKS_TABLE,
    WINDOW_SIZE, MIN_SENTENCE_LEN,
    TARGET_PARAGRAPH_TOKENS, PARAGRAPH_OVERLAP_TOKENS, APPROX_CHARS_PER_TOKEN,
    SUMMARY_LLM_ENDPOINT, SECTION_SUMMARY_MAX_TOKENS, DOC_SUMMARY_MAX_TOKENS,
)

# Reuse the section parsing & cleaning helpers from the existing chunker
from chunker import (
    clean_jd_text, parse_sections, split_sentences, get_section_type,
)


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED CHUNK PRODUCTION (sentence + paragraph)
#
# A pandas UDF processes each doc, returns a list of chunk structs at both
# sentence and paragraph granularity. Section and document summaries are
# produced by ai_query in the SQL layer below — keeping LLM calls out of the
# UDF lets Databricks batch them.
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_CHUNK_SCHEMA = ArrayType(StructType([
    StructField("chunk_id",        StringType(), False),
    StructField("granularity",     StringType(), False),
    StructField("parent_chunk_id", StringType(), True),
    StructField("sentence",        StringType(), False),  # what we embed
    StructField("window_text",     StringType(), False),  # what we display
    StructField("section",         StringType(), False),
    StructField("section_type",    StringType(), False),
    StructField("sentence_index",  StringType(), False),
    StructField("token_count_est", IntegerType(), False),
]))


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // APPROX_CHARS_PER_TOKEN)


def _pack_paragraphs(sentences: list,
                     target_tokens: int = TARGET_PARAGRAPH_TOKENS,
                     overlap_tokens: int = PARAGRAPH_OVERLAP_TOKENS) -> list:
    """
    Greedy sentence-aware packer. Each output paragraph is a list of
    consecutive sentences whose combined token estimate is close to
    target_tokens. The next paragraph starts with up to overlap_tokens
    of trailing sentences from the previous one — this preserves
    cross-paragraph context without heavy duplication.

    Returns a list of (sentence_indexes, joined_text) tuples.
    """
    if not sentences:
        return []

    paragraphs = []
    current_sents:  list = []
    current_idxs:   list = []
    current_tokens: int  = 0

    for i, sent in enumerate(sentences):
        st = _approx_tokens(sent)
        if current_tokens + st > target_tokens and current_sents:
            paragraphs.append((list(current_idxs), " ".join(current_sents)))

            # Build overlap from the tail of the just-emitted paragraph.
            overlap_sents:  list = []
            overlap_idxs:   list = []
            overlap_tok:    int  = 0
            for s, idx in zip(reversed(current_sents), reversed(current_idxs)):
                t = _approx_tokens(s)
                if overlap_tok + t > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_idxs.insert(0, idx)
                overlap_tok += t

            current_sents  = overlap_sents + [sent]
            current_idxs   = overlap_idxs  + [i]
            current_tokens = overlap_tok + st
        else:
            current_sents.append(sent)
            current_idxs.append(i)
            current_tokens += st

    if current_sents:
        paragraphs.append((list(current_idxs), " ".join(current_sents)))
    return paragraphs


@F.pandas_udf(LOCAL_CHUNK_SCHEMA)
def multi_granularity_local_udf(
    doc_ids: pd.Series, texts: pd.Series, org_keys: pd.Series
) -> pd.Series:
    """
    Produces sentence-level and paragraph-level chunks per doc.
    Section and document summaries are added later by SQL ai_query.

    parent_chunk_id wiring at this stage:
        sentence  → its enclosing paragraph chunk_id
        paragraph → its section's section-summary chunk_id (deterministic id,
                    even though the section_summary row is written later)
    """
    out = []
    for doc_id, text, org_key in zip(doc_ids, texts, org_keys):
        org_key = (org_key or "").lower()
        text    = clean_jd_text(text, org_key)
        if not text:
            out.append([])
            continue

        sections = parse_sections(text, org_key)
        doc_chunks: list = []

        for sec_idx, (header, body) in enumerate(sections.items()):
            sentences = split_sentences(body)
            if not sentences:
                continue
            sec_type    = get_section_type(header)
            safe_header = re.sub(r"[^\w]", "_", header)[:30]

            # Deterministic id for this section's summary (the row itself is
            # produced later by ai_query — we emit the id now so paragraphs
            # can point at it).
            section_summary_id = f"{doc_id}__{safe_header}__section_summary"

            # ── paragraph chunks for this section ───────────────────
            paragraphs = _pack_paragraphs(sentences)
            paragraph_ids_by_sentence: dict = {}

            for p_idx, (sent_idxs, joined) in enumerate(paragraphs):
                para_chunk_id = f"{doc_id}__{safe_header}__p{p_idx}"
                doc_chunks.append({
                    "chunk_id":        para_chunk_id,
                    "granularity":     "paragraph",
                    "parent_chunk_id": section_summary_id,
                    "sentence":        joined,        # embedded
                    "window_text":     joined,        # displayed
                    "section":         header,
                    "section_type":    sec_type,
                    "sentence_index":  f"p{p_idx}",
                    "token_count_est": _approx_tokens(joined),
                })
                for sidx in sent_idxs:
                    paragraph_ids_by_sentence[sidx] = para_chunk_id

            # ── sentence chunks (windowed display, atomic embedding) ──
            for i, sentence in enumerate(sentences):
                w_start = max(0, i - WINDOW_SIZE)
                w_end   = min(len(sentences), i + WINDOW_SIZE + 1)
                window  = sentences[w_start:w_end]
                marked  = [
                    f">>> {s} <<<" if (w_start + j) == i else s
                    for j, s in enumerate(window)
                ]
                sent_chunk_id = f"{doc_id}__{safe_header}__s{i}"
                doc_chunks.append({
                    "chunk_id":        sent_chunk_id,
                    "granularity":     "sentence",
                    # parent = the paragraph this sentence ended up in
                    "parent_chunk_id": paragraph_ids_by_sentence.get(
                        i, section_summary_id),
                    "sentence":        sentence,
                    "window_text":     " ".join(marked),
                    "section":         header,
                    "section_type":    sec_type,
                    "sentence_index":  f"s{i}",
                    "token_count_est": _approx_tokens(sentence),
                })

        out.append(doc_chunks)
    return pd.Series(out)


# ─────────────────────────────────────────────────────────────────────────────
# JOB-LEVEL COLUMN LIST
#
# Every chunk row carries the same set of doc-level filter columns so the
# retriever can filter without joining back to clean_openings. We define this
# once and reuse for all four granularities.
# ─────────────────────────────────────────────────────────────────────────────

JOB_LEVEL_COLS = [
    "doc_id", "source_type", "source_url",
    "org_key", "job_id",
    "title_clean", "job_family", "domain",
    "team_name", "seniority", "seniority_source",
    "is_intern", "job_location_clean", "country",
    "file_dt", "notional_job_posted_dt",
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION SUMMARIES (one row per (doc, section))
#
# We use ai_query at SQL level so Databricks Model Serving can batch the calls.
# Input = paragraph rows for the section (already cleaned, deduplicated).
# Output = one section_summary chunk pointing at the document_summary chunk.
# ─────────────────────────────────────────────────────────────────────────────

SECTION_SUMMARY_PROMPT = (
    "Summarize the following section of a job posting in 2-4 sentences. "
    "Capture the core facts only: required skills, technologies, years of "
    "experience, scope of responsibility, team focus, or location specifics, "
    "depending on the section type. Do not invent. Do not include disclaimers. "
    "Return only the summary text.\\n\\nSection text:\\n"
)

DOC_SUMMARY_PROMPT = (
    "You are summarizing a job posting for downstream retrieval and analytics. "
    "Write 3-5 sentences capturing: (1) the role and seniority, (2) the team "
    "or product area, (3) the most distinctive required skills/tools, "
    "(4) location or work arrangement if stated, and (5) anything that makes "
    "this posting unusual. Do not invent facts. Return only the summary.\\n\\n"
    "Section summaries:\\n"
)


def build_section_summaries(spark, paragraph_df: DataFrame) -> DataFrame:
    """
    For each (doc_id, section), concatenate its paragraph chunks and ask
    ai_query for a short summary. Returns a DataFrame with the same chunk
    schema as the rule-based output, granularity='section'.
    """
    # Concatenate all paragraph text per (doc_id, section, section_type)
    grouped = (
        paragraph_df
        .groupBy("doc_id", "section", "section_type")
        .agg(
            F.concat_ws("\n\n", F.collect_list("window_text")).alias("section_text"),
        )
        .filter(F.length("section_text") >= 80)
    )

    # Use ai_query at SQL level — Databricks batches under the hood
    grouped.createOrReplaceTempView("_section_inputs")
    summarized = spark.sql(f"""
        SELECT
            doc_id,
            section,
            section_type,
            ai_query(
                '{SUMMARY_LLM_ENDPOINT}',
                CONCAT('{SECTION_SUMMARY_PROMPT}', section_text),
                modelParameters => named_struct(
                    'max_tokens',  {SECTION_SUMMARY_MAX_TOKENS},
                    'temperature', 0.0
                )
            ) AS summary_text
        FROM _section_inputs
    """)

    # Re-shape into the chunk schema and join job-level metadata
    job_level_df = paragraph_df.select(
        "doc_id", *[c for c in JOB_LEVEL_COLS if c != "doc_id"]
    ).dropDuplicates(["doc_id"])

    enriched = (
        summarized
        .withColumn("safe_header",
                    F.regexp_replace(F.substring(F.col("section"), 1, 30),
                                     r"[^\w]", "_"))
        .withColumn("chunk_id",
                    F.concat_ws("__", "doc_id", "safe_header",
                                F.lit("section_summary")))
        .withColumn("granularity",     F.lit("section"))
        .withColumn("parent_chunk_id", F.concat_ws("__",
                                                    "doc_id",
                                                    F.lit("doc_summary")))
        .withColumnRenamed("summary_text", "sentence")
        .withColumn("window_text",     F.col("sentence"))
        .withColumn("sentence_index",  F.lit("section_summary"))
        .withColumn("token_count_est",
                    (F.length("sentence") / APPROX_CHARS_PER_TOKEN).cast("int"))
        .drop("safe_header")
        .filter(F.col("sentence").isNotNull())
        .filter(F.length("sentence") > 20)
        .join(job_level_df, on="doc_id", how="left")
    )
    return enriched


def build_document_summaries(spark, section_summary_df: DataFrame) -> DataFrame:
    """
    For each doc, concatenate its section summaries and ask ai_query for a
    document-level summary. Returns DataFrame with granularity='document'.
    """
    grouped = (
        section_summary_df
        .groupBy("doc_id")
        .agg(
            F.concat_ws("\n",
                        F.collect_list(F.concat_ws(": ", "section", "sentence"))
                        ).alias("doc_text"),
        )
        .filter(F.length("doc_text") >= 80)
    )

    grouped.createOrReplaceTempView("_doc_inputs")
    summarized = spark.sql(f"""
        SELECT
            doc_id,
            ai_query(
                '{SUMMARY_LLM_ENDPOINT}',
                CONCAT('{DOC_SUMMARY_PROMPT}', doc_text),
                modelParameters => named_struct(
                    'max_tokens',  {DOC_SUMMARY_MAX_TOKENS},
                    'temperature', 0.0
                )
            ) AS summary_text
        FROM _doc_inputs
    """)

    job_level_df = section_summary_df.select(
        "doc_id", *[c for c in JOB_LEVEL_COLS if c != "doc_id"]
    ).dropDuplicates(["doc_id"])

    enriched = (
        summarized
        .withColumn("chunk_id",
                    F.concat_ws("__", "doc_id", F.lit("doc_summary")))
        .withColumn("granularity",     F.lit("document"))
        .withColumn("parent_chunk_id", F.lit(None).cast("string"))
        .withColumn("section",         F.lit("(whole document)"))
        .withColumn("section_type",    F.lit("document_summary"))
        .withColumnRenamed("summary_text", "sentence")
        .withColumn("window_text",     F.col("sentence"))
        .withColumn("sentence_index",  F.lit("doc_summary"))
        .withColumn("token_count_est",
                    (F.length("sentence") / APPROX_CHARS_PER_TOKEN).cast("int"))
        .filter(F.col("sentence").isNotNull())
        .filter(F.length("sentence") > 20)
        .join(job_level_df, on="doc_id", how="left")
    )
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL DRIVER
#
# Called from chunker.py main() when CHUNKING_STRATEGY == "multi_granularity".
# Returns a DataFrame matching the CHUNKS_TABLE schema, ready to MERGE.
# ─────────────────────────────────────────────────────────────────────────────

def build_multi_granularity_chunks(spark, df_to_chunk: DataFrame) -> DataFrame:
    """
    Input: df_to_chunk   = a slice of clean_openings (already watermark-bounded)
    Output: DataFrame with one row per chunk (4 granularities), ready to MERGE
            into CHUNKS_TABLE.
    """
    # ── 1. sentence + paragraph chunks (rule-based) ────────────────
    df_local = (
        df_to_chunk
        .withColumn("chunks", multi_granularity_local_udf(
            F.col("doc_id"),
            F.col("job_description"),
            F.col("org_key"),
        ))
        .withColumn("chunk", F.explode_outer("chunks"))
        .filter(F.col("chunk").isNotNull())
        .select(
            F.col("chunk.chunk_id").alias("chunk_id"),
            F.col("chunk.granularity").alias("granularity"),
            F.col("chunk.parent_chunk_id").alias("parent_chunk_id"),
            F.col("chunk.sentence").alias("sentence"),
            F.col("chunk.window_text").alias("window_text"),
            F.col("chunk.section").alias("section"),
            F.col("chunk.section_type").alias("section_type"),
            F.col("chunk.sentence_index").alias("sentence_index"),
            F.col("chunk.token_count_est").alias("token_count_est"),
            *JOB_LEVEL_COLS,
        )
    )

    # Materialize once — the section/document stages read this twice.
    df_local.cache()
    _ = df_local.count()  # force cache

    paragraph_df = df_local.filter(F.col("granularity") == "paragraph")

    # ── 2. section summaries (LLM via ai_query) ────────────────────
    section_df = build_section_summaries(spark, paragraph_df)

    # ── 3. document summaries (LLM via ai_query) ───────────────────
    document_df = build_document_summaries(spark, section_df)

    # ── 4. union all four granularities ────────────────────────────
    # Align column order across the three sources before union.
    final_cols = [
        "chunk_id", "granularity", "parent_chunk_id",
        "sentence", "window_text",
        "section", "section_type", "sentence_index", "token_count_est",
        *JOB_LEVEL_COLS,
    ]

    unified = (
        df_local.select(*final_cols)
        .unionByName(section_df.select(*final_cols))
        .unionByName(document_df.select(*final_cols))
    )

    # trend_dt is computed downstream the same way as before
    unified = unified.withColumn(
        "trend_dt",
        F.coalesce(
            F.to_date(F.col("notional_job_posted_dt")),
            F.to_date(F.col("file_dt")),
        ),
    )
    return unified
