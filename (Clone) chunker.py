# Databricks notebook source
# ══════════════════════════════════════════════════════════════════════════════
# 04_chunker.py
#
# Sentence window chunking with org-aware section detection.
# Watermark design: uses _pipeline stage (not per-org) because chunking
# processes all orgs together from clean_openings.
#
# Gate logic:
#   upper bound = get_pipeline_watermark() = MIN(normalize watermarks across orgs)
#   lower bound = get_stage_watermark("chunk")
#   window = (lower, upper]
#
# This ensures chunking never gets ahead of the slowest org at normalization,
# and adding a new org (Microsoft) automatically tightens the upper bound.
# ══════════════════════════════════════════════════════════════════════════════

# COMMAND ----------

from pipeline_watermark import WatermarkManager
from properties import (
    CLEAN_TABLE, CHUNKS_TABLE,
    WINDOW_SIZE, MIN_SENTENCE_LEN,
    CHUNKING_STRATEGY,
)

import os
import re
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StructType, StructField, StringType


# ─────────────────────────────────────────────────────────────
# SECTION HEADER PATTERNS
# ─────────────────────────────────────────────────────────────

GOOGLE_SECTION_RE = re.compile(
    r"(?im)^(about the (?:job|role|team)|"
    r"minimum qualifications?:?|preferred qualifications?:?|"
    r"responsibilities:?)",
)
AMAZON_SECTION_RE = re.compile(
    r"(?im)^(description|about amazon|about the role|about the team|"
    r"basic qualifications?:?|required qualifications?:?|"
    r"preferred qualifications?:?|"
    r"key job responsibilities?:?|core responsibilities?:?|"
    r"what you.?ll do:?|what we offer:?)",
)
SECTION_TYPE_MAP = {
    "description":              "about",
    "about amazon":             "about",
    "about the job":            "about",
    "about the role":           "about",
    "about the team":           "about",
    "minimum qualifications":   "qualifications_required",
    "basic qualifications":     "qualifications_required",
    "required qualifications":  "qualifications_required",
    "preferred qualifications": "qualifications_preferred",
    "responsibilities":         "responsibilities",
    "key job responsibilities":  "responsibilities",
    "core responsibilities":    "responsibilities",
    "what you'll do":           "responsibilities",
    "what you will do":         "responsibilities",
}

def get_section_type(header: str) -> str:
    h = header.lower().strip().rstrip(":")
    for key, val in SECTION_TYPE_MAP.items():
        if key in h:
            return val
    return "other"


# ─────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────

# --- Generic boilerplate patterns (apply to ALL orgs) ---

# EEO: handles "is an equal opportunity employer", "is proud to be an equal
# opportunity employer", "we are an equal opportunity employer", etc.
GENERIC_EEO_RE = re.compile(
    r"(?:we\s+are|is(?:\s+proud\s+to\s+be)?)\s+an?\s+equal\s+opportunity\s+employer[^.]*\.",
    re.IGNORECASE | re.DOTALL,
)

GENERIC_DISABILITY_RE = re.compile(
    r"(?:if you (?:have|need|require) a disability|"
    r"(?:is\s+)?(?:also\s+)?committed\s+to\s+providing\s+reasonable\s+accommodations?|"
    r"reasonable\s+accommodations?\s+(?:for|to|will be)|"
    r"individuals?\s+with\s+disabilities?\s+(?:are|who|may)|"
    r"if you need (?:assistance|an accommodation)\s+due\s+to\s+a\s+disability)"
    r"[^.]*\.",
    re.IGNORECASE | re.DOTALL,
)

# Salary: handles "Base Salary:\n$X - $Y", "$X-$Y per year", and
# "the salary range for this role is..." formats
GENERIC_SALARY_RE = re.compile(
    r"^\s*(?:the\s+)?(?:base\s+)?(?:salary|pay|compensation)\s*(?:range|for this (?:role|position))?:?\s*$|"
    r"^\s*\$[\d,]+(?:\s*[-–]\s*\$[\d,]+)?\.?.*?(?:per\s+(?:year|annum|hour)|annually|/yr|"
    r"(?:the\s+)?final\s+compensation|will\s+be\s+determined|based\s+on\s+(?:your\s+)?experience).*$",
    re.IGNORECASE | re.MULTILINE,
)

GENERIC_BENEFITS_RE = re.compile(
    r"(?:learn more about (?:our )?benefits|"
    r"for (?:a )?(?:full|complete) (?:list|description) of (?:our )?benefits|"
    r"benefits (?:include|package|overview):?)"
    r"[^.]*\.",
    re.IGNORECASE | re.DOTALL,
)

GENERIC_APPLY_CTA_RE = re.compile(
    r"^\s*(?:apply\s+(?:now|today|here)|click\s+(?:here|below)\s+to\s+apply|"
    r"ready\s+to\s+(?:apply|join)|submit\s+your\s+(?:application|resume))[^.]*\.?$",
    re.IGNORECASE | re.MULTILINE,
)

# Non-discrimination: handles "We do not discriminate in hiring...",
# "...on the basis of...", "without regard to..."
GENERIC_LEGAL_RE = re.compile(
    r"(?:we\s+do\s+not\s+discriminate\s+(?:in\s+\w+\s+)?(?:(?:on\s+the\s+)?basis\s+of\s+|based\s+on\s+)?|"
    r"all\s+qualified\s+applicants\s+will\s+receive\s+consideration|"
    r"employment\s+decisions?\s+(?:are|will be)\s+made\s+without\s+regard\s+to)"
    r"[^.]*\.",
    re.IGNORECASE | re.DOTALL,
)

# Diversity encouragement: "We hire from diverse backgrounds...encourage you to apply"
GENERIC_DIVERSITY_ENCOURAGE_RE = re.compile(
    r"(?:we\s+hire\s+(?:talented\s+)?(?:and\s+passionate\s+)?people\s+from\s+(?:a\s+)?(?:variety|wide\s+range)\s+of\s+backgrounds|"
    r"if\s+you(?:'re|.re)\s+excited\s+about\s+(?:a\s+|this\s+)?role\s+but\s+your\s+(?:past\s+)?experience\s+doesn(?:'t|.t)\s+align)"
    r"[^.]*\.(?:\s+[^.]*(?:encourage\s+you\s+to\s+apply|we\s+(?:still\s+)?(?:want\s+to\s+hear|encourage)[^.]*)\.)?"
    r"(?:\s+[^.]*(?:we\s+want\s+to\s+hear\s+from\s+you|encourage\s+you\s+to\s+apply)[^.]*\.)?",
    re.IGNORECASE | re.DOTALL,
)

# Criminal history: "considers qualified applicants with criminal histories"
GENERIC_CRIMINAL_HISTORY_RE = re.compile(
    r"[^.]*(?:considers?\s+qualified\s+applicants?\s+with\s+criminal\s+histor(?:ies|y)|"
    r"consistent\s+with\s+applicable\s+(?:federal,?\s*)?(?:state,?\s*)?(?:and\s+)?local\s+law)[^.]*\.",
    re.IGNORECASE | re.DOTALL,
)

# Privacy/consent: "By clicking Submit Application, I agree..."
GENERIC_PRIVACY_CONSENT_RE = re.compile(
    r"(?:by\s+clicking|by\s+submitting)[^.]*(?:privacy\s+policy|data\s+(?:processing|protection)|"
    r"will\s+collect\s+and\s+process)[^.]*\.",
    re.IGNORECASE | re.DOTALL,
)

# Hashtag labels: #LI-Onsite, #LI-Remote, #LI-Hybrid, etc.
GENERIC_HASHTAG_LABEL_RE = re.compile(
    r"^\s*#[A-Z]{2,}-\w+\s*$",
    re.MULTILINE,
)

# --- Amazon-specific boilerplate (additional layer) ---

AMAZON_BOILERPLATE_RE = re.compile(
    r"Our inclusive culture empowers Amazonians.*?please contact your Recruiting Partner\.|"
    r"Amazon is an equal opportunity employer.*?protected status\.|"
    r"If you have a disability.*?please visit.*?for more information\.|"
    r"Learn more about our benefits at.*?\.",
    re.IGNORECASE | re.DOTALL
)


def clean_jd_text(text: str, org_key: str) -> str:
    """
    Clean job description text. Applies generic boilerplate removal for ALL
    orgs, then org-specific patterns where available.
    """
    if not text or str(text).lower() in ("null", "none"):
        return ""
    t = str(text).strip()

    # --- Generic cleaning (all orgs) ---
    t = GENERIC_EEO_RE.sub("", t)
    t = GENERIC_DISABILITY_RE.sub("", t)
    t = GENERIC_SALARY_RE.sub("", t)
    t = GENERIC_BENEFITS_RE.sub("", t)
    t = GENERIC_APPLY_CTA_RE.sub("", t)
    t = GENERIC_LEGAL_RE.sub("", t)
    t = GENERIC_DIVERSITY_ENCOURAGE_RE.sub("", t)
    t = GENERIC_CRIMINAL_HISTORY_RE.sub("", t)
    t = GENERIC_PRIVACY_CONSENT_RE.sub("", t)
    t = GENERIC_HASHTAG_LABEL_RE.sub("", t)
    t = re.sub(r"https?://\S+", "", t)                        # URLs
    t = re.sub(r"^\s*[-•]\s*", "", t, flags=re.MULTILINE)     # bullet markers
    t = re.sub(r"^\s*#{1,3}\s*$", "", t, flags=re.MULTILINE)  # stray markdown headers

    # --- Amazon-specific cleaning (additional layer) ---
    if "amazon" in (org_key or "").lower():
        t = AMAZON_BOILERPLATE_RE.sub("", t)
        t = re.sub(r"^\s*\$[\d,]+.*?annually.*$", "", t, flags=re.MULTILINE | re.IGNORECASE)

    # --- Final whitespace normalization ---
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r" {2,}", " ", t)
    return t.strip()

def parse_sections(text: str, org_key: str) -> dict:
    pattern = AMAZON_SECTION_RE if "amazon" in (org_key or "").lower() \
              else GOOGLE_SECTION_RE
    matches = list(pattern.finditer(text))
    if len(matches) < 2:
        return {"full description": text}
    sections = {}
    for i, match in enumerate(matches):
        header = match.group(1).strip().rstrip(":").lower()
        start  = match.end()
        end    = matches[i+1].start() if i+1 < len(matches) else len(text)
        body   = text[start:end].strip()
        if body and len(body) > 30 and header not in sections:
            sections[header] = body
    return sections

def split_sentences(text: str) -> list:
    raw       = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) < MIN_SENTENCE_LEN:
            continue
        s = re.sub(r"^[\-•\*]\s+", "", s)
        s = re.sub(r"^\d+\.\s+", "", s)
        if len(s) >= MIN_SENTENCE_LEN:
            sentences.append(s)
    return sentences


# ─────────────────────────────────────────────────────────────
# SENTENCE WINDOW CHUNKING UDF
# ─────────────────────────────────────────────────────────────

CHUNK_SCHEMA = ArrayType(StructType([
    StructField("chunk_id",       StringType(), False),
    StructField("sentence",       StringType(), False),
    StructField("window_text",    StringType(), False),
    StructField("section",        StringType(), False),
    StructField("section_type",   StringType(), False),
    StructField("sentence_index", StringType(), False),
]))

@F.pandas_udf(CHUNK_SCHEMA)
def sentence_window_chunk_udf(
    doc_ids: pd.Series, texts: pd.Series, org_keys: pd.Series
) -> pd.Series:
    """
    For each sentence S[i]:
      sentence    = S[i]                                → what gets embedded
      window_text = S[i-W] ... >>>S[i]<<< ... S[i+W]   → what gets sent to LLM
    """
    all_results = []
    for doc_id, text, org_key in zip(doc_ids, texts, org_keys):
        org_key  = (org_key or "").lower()
        text     = clean_jd_text(text, org_key)
        if not text:
            all_results.append([])
            continue

        sections   = parse_sections(text, org_key)
        doc_chunks = []

        for header, body in sections.items():
            sentences = split_sentences(body)
            if not sentences:
                continue
            sec_type = get_section_type(header)

            for i, sentence in enumerate(sentences):
                w_start = max(0, i - WINDOW_SIZE)
                w_end   = min(len(sentences), i + WINDOW_SIZE + 1)
                window  = sentences[w_start:w_end]
                marked  = [
                    f">>> {s} <<<" if (w_start + j) == i else s
                    for j, s in enumerate(window)
                ]
                safe_header = re.sub(r"[^\w]", "_", header)[:30]
                doc_chunks.append({
                    "chunk_id":       f"{doc_id}__{safe_header}__{i}",
                    "sentence":       sentence,
                    "window_text":    " ".join(marked),
                    "section":        header,
                    "section_type":   sec_type,
                    "sentence_index": str(i),
                })

        all_results.append(doc_chunks)
    return pd.Series(all_results)


# ─────────────────────────────────────────────────────────────
# APPLY CHUNKING
# ─────────────────────────────────────────────────────────────

# COMMAND ----------

def _apply_sentence_window_strategy(df_to_chunk):
    """Original sentence-window chunking. One row per sentence, with
    ±WINDOW_SIZE display context. Kept for backwards compatibility and
    A/B comparison with the multi-granularity strategy."""
    return (
        df_to_chunk
        .withColumn("chunks", sentence_window_chunk_udf(
            F.col("doc_id"),
            F.col("job_description"),
            F.col("org_key"),
        ))
        .withColumn("chunk", F.explode("chunks"))
        .select(
            F.col("chunk.chunk_id").alias("chunk_id"),
            F.lit("sentence").alias("granularity"),
            F.lit(None).cast("string").alias("parent_chunk_id"),
            F.col("chunk.sentence").alias("sentence"),
            F.col("chunk.window_text").alias("window_text"),
            F.col("chunk.section").alias("section"),
            F.col("chunk.section_type").alias("section_type"),
            F.col("chunk.sentence_index").alias("sentence_index"),
            (F.length(F.col("chunk.sentence")) / 4).cast("int").alias("token_count_est"),
            "doc_id","source_type","source_url",
            "org_key","job_id",
            "title_clean","job_family","domain",
            "team_name","seniority","seniority_source",
            "is_intern","job_location_clean","country",
            "file_dt","notional_job_posted_dt",
        )
        .withColumn("trend_dt",
            F.coalesce(
                F.to_date(F.col("notional_job_posted_dt")),
                F.to_date(F.col("file_dt"))
            )
        )
    )


def apply_chnunking(wm: WatermarkManager, normalize_floor: str, chunk_wm: str)->bool:
    try:
        df_to_chunk = (
            spark.table(CLEAN_TABLE)
            .filter(F.col("file_dt") >  F.lit(chunk_wm))
            .filter(F.col("file_dt") <= F.lit(normalize_floor))
            # No org filter — all orgs processed together
        )

        chunk_input_count = df_to_chunk.count()
        print(f"Documents to chunk: {chunk_input_count}")
        print(f"Chunking strategy: {CHUNKING_STRATEGY}")

        if chunk_input_count == 0:
            wm.mark_stage_success("chunk", normalize_floor, 0)
            print("No data to chunk, exiting.")
            return True

        # ── Strategy dispatch ──────────────────────────────────────
        if CHUNKING_STRATEGY == "multi_granularity":
            # Lazy import keeps the sentence_window-only path free of any
            # transitive dependency on chunking_strategies.py.
            from chunking_strategies import build_multi_granularity_chunks
            df_chunked = build_multi_granularity_chunks(spark, df_to_chunk)
        else:
            df_chunked = _apply_sentence_window_strategy(df_to_chunk)

        # MERGE on (doc_id, chunk_id) — idempotent across re-runs
        df_chunked.createOrReplaceTempView("chunks_source")
        spark.sql(f"""
            MERGE INTO {CHUNKS_TABLE} AS T
            USING chunks_source AS S
            ON T.doc_id = S.doc_id AND T.chunk_id = S.chunk_id
            WHEN MATCHED     THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """)

        total_chunks = df_chunked.count()
        print(f"Chunks written: {total_chunks}")

        # Per-granularity counts make it obvious if the LLM-summary stage
        # silently produced zero rows (e.g. ai_query throttling).
        try:
            df_chunked.groupBy("granularity").count() \
                      .orderBy("granularity").show(truncate=False)
        except Exception:
            pass

        wm.mark_stage_success("chunk", normalize_floor, total_chunks)

    except Exception as e:
        wm.mark_stage_failed("chunk", str(e))
        return False
    return True





# ─────────────────────────────────────────────────────────────
# WATERMARK GATE
# ─────────────────────────────────────────────────────────────
# # Upper bound: what has normalize confirmed for ALL orgs?
# normalize_floor = wm.get_pipeline_watermark()
# # Lower bound: what has chunk already processed?
# chunk_wm        = wm.get_stage_watermark("chunk")

def main()->None:
    wm=WatermarkManager(spark)
    # Multi-granularity introduces new columns (granularity, parent_chunk_id,
    # token_count_est). Allow Delta MERGE to add them on first run rather
    # than forcing a manual ALTER TABLE.
    spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")
    wm.register_org("_pipeline", stage="chunk", seed_dt="2024-01-01")
    normalize_floor = wm.get_pipeline_watermark()
    chunk_wm        = wm.get_stage_watermark("chunk")
    if normalize_floor is None:
        print("No orgs have normalized yet — nothing to chunk")
        return None
    if chunk_wm is not None and chunk_wm >= normalize_floor:
        print(f"Chunk watermark {chunk_wm} is current ({normalize_floor}) — nothing to do")
        return None
    if apply_chnunking(wm, normalize_floor, chunk_wm):
        print("Chunking complete")
    else:
        print("Chunking failed")  
    wm.status()  
    return None    

if __name__ == "__main__":
    main()
    
