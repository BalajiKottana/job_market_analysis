# Databricks notebook source
# ══════════════════════════════════════════════════════════════════════════════
# 03_data_cleaning.py
#
# Watermark-gated normalization pipeline.
#
# Key decisions:
#   - REQUIRE_ALL_ORGS=True: holds until ALL registered orgs have new data
#     (keeps agent dataset balanced across sources)
#   - Filters raw_openings to (last_wm, max_new_dt] per org — never reprocesses
#   - clean_openings write uses MERGE ON doc_id — idempotent, no duplicates
#   - mark_success advances watermark ONLY after full success
#   - mark_failed leaves date unchanged → automatic retry next run
#   - Failed orgs block the run; no_data orgs do not (they advanced their date)
# ══════════════════════════════════════════════════════════════════════════════

# COMMAND ----------

import os
from pipeline_watermark import WatermarkManager
import re
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
import pandas as pd
from properties import RAW_TABLE, CLEAN_TABLE, QUARANTINE_TABLE

# Set True → wait until ALL registered orgs have new data (recommended)
# Set False → normalize whichever orgs are ready independently
REQUIRE_ALL_ORGS = True


# ─────────────────────────────────────────────────────────────
# STEP 1 — TITLE CLEANING UDF
# Strips AMZ req IDs, unicode brackets, year prefixes,
# internship country suffixes, duplicate team names
# ─────────────────────────────────────────────────────────────

def clean_title(title: str) -> str:
    if not title:
        return ""
    t = title.strip()
    t = re.sub(r"【[^】]*】", "", t)
    t = re.sub(r"[【】『』「」《》〔〕]", "", t)
    t = re.sub(r"\s*[-–]\s*AMZ[\w.]+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\d{4}\s+", "", t)
    t = re.sub(
        r"\s*[-–,]\s*(?:France|UK|Esp|Lux|US|USA|EMEA|LATAM|APAC|Brazil|"
        r"Germany|Japan|China|India|Australia|Canada)\b.*$",
        "", t, flags=re.IGNORECASE
    )
    t = re.sub(r",?\s*Amazon University.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*\(US\)\s*$", "", t)
    t = re.sub(r",\s*([^,]+),\s*\1\s*$", r", \1", t, flags=re.IGNORECASE)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+,", ",", t)
    return t.strip().strip(",").strip()


# ─────────────────────────────────────────────────────────────
# STEP 2 — JOB FAMILY CLASSIFIER
# Data Center Operations separated from Data Engineering (order matters)
# ─────────────────────────────────────────────────────────────

JOB_FAMILY_RULES = [
    (r"data cent(?:er|re)\s+(?:chief|controls|facility|operations|"
     r"infrastructure|structural|it support|linux|project|technical ops?|"
     r"engineering ops?|commissioning|regional|night shift|hw engineer|"
     r"hardware|mechanical|electrical|energy|power|security manager|quality|"
     r"technician|controls systems|design architect|plant engineer|area lead|"
     r"services manager|strategic negotiator)|"
     r"\bdceo\b|dcc communities|aws data cent(?:er|re)|"
     r"data centre linux|data center ops engineer|server engineer.*data cent|"
     r"systems automation.*data cent|physical security.*data cent|"
     r"senior material engineer.*data cent|"
     r"critical infrastructure.*(?:sr\.?\s+)?mechanical.*data cent|"
     r"connectivity engineer.*data cent|regional safety.*data cent|"
     r"cloud hardware.*aws.*data cent|commissioning engineer.*data cent|"
     r"data center technician|data centre technician|"
     r"data center facilities|data centre facilities|"
     r"data center manager|data center operations manager|"
     r"data center server operations|data center area lead|"
     r"data center design|data center mechanical|data center electrical|"
     r"data center plant engineer|data center equipment|"
     r"data center construction|data center delivery|"
     r"data center controls|data center quality|data center security",
     "Data Center Operations"),
    (r"\binternship\b|\bintern\b|co-op|class of 20\d\d|apprenticeship",
     "Internship"),
    (r"\bai platform data engineer\b",
     "AI Platform Data Engineering"),
    (r"data engineering manager|manager.*data engineering|"
     r"youtube manager.*data engineering",
     "Data Engineering Management"),
    (r"(?:sr\.?|senior|principal)\s+data infrastructure engineer",
     "Data Engineering"),
    (r"^(?:(?:sr\.?|senior|principal|lead|staff|cloud)\s+)?"
     r"(?:big\s+)?data (?:engineer|infrastructure engineer)"
     r"(?:\s+(?:i{1,3}|ii|iii))?(?:\s*$|[\s,\-])",
     "Data Engineering"),
    (r"data engineer.*manager|engineering manager.*data",
     "Data Engineering Management"),
    (r"^(?:(?:sr\.?|senior|principal|staff|lead|business|product|"
     r"research|zappos)\s+)?data scientist",
     "Data Science"),
    (r"\bapplied scientist\b|\bresearch scientist\b",
     "Applied Science"),
    (r"\bmachine learning engineer\b|\bml engineer\b|"
     r"machine learning compiler|machine learning performance|"
     r"machine learning scientist",
     "ML Engineering"),
    (r"\bagentic ai\b|\bgenai\b|\bgenerative ai\b|"
     r"\bai (?:platform|product|content|data|benchmarking|compliance|"
     r"empowerment|editorial|language|process|tutor|trainer|innovation|"
     r"hardware systems)\b|deep learning architect|"
     r"(?:sr\.?|senior|principal)?\s+ai (?:specialist|solution architect|"
     r"ml consultant|sales)",
     "AI/GenAI"),
    (r"\bbie\b|business intelligence engineer",
     "Business Intelligence"),
    (r"\bdata analyst\b|\bfinance.*analyst\b|\bfinancial analyst\b|"
     r"\bbusiness analyst\b|trans planning.*analyst|"
     r"capacity planner|production planning analyst",
     "Data Analytics"),
    (r"software dev(?:elopment)? (?:engineer|manager)|\bsde\b|\bsdm\b|\bsoftware engineer\b",
     "Software Engineering"),
    (r"solutions architect|delivery consultant|specialist.*architect|delivery practice manager",
     "Solutions Architecture"),
    (r"technical program manager|\btpm\b|product manager.*technical|\bprincipal pmt\b",
     "Technical Program/Product"),
]

def classify_job_family(title: str) -> str:
    if not title:
        return "Other"
    t = title.lower().strip()
    for pattern, family in JOB_FAMILY_RULES:
        if re.search(pattern, t, re.IGNORECASE):
            return family
    return "Other"


# ─────────────────────────────────────────────────────────────
# STEP 3 — SENIORITY EXTRACTOR
# Google: job_position field PRIMARY (Early/Mid/Advanced), enriched from title
# Amazon: title-only (job_position always null)
# ─────────────────────────────────────────────────────────────

def extract_seniority(title: str, job_position: str, org_key: str) -> tuple:
    tl        = (title or "").lower().strip()
    jp        = (job_position or "").strip().lower()
    null_jp   = jp in ("null", "none", "")
    is_google = (org_key or "").lower() in ("google", "youtube", "deepmind")

    # Intern — all orgs, highest priority
    if re.search(r"\binternship\b|\bintern\b|co-op|class of 20\d\d|"
                 r"apprenticeship|next step.*intern", tl):
        return ("Intern", "title_keyword")
    if re.search(r"^\d{4}\s", tl):
        return ("Intern", "year_prefix")
    if re.search(r"post.?doctoral|post.?doc", tl):
        return ("Post-Doc", "title_keyword")

    if is_google and not null_jp:
        is_staff        = bool(re.search(r"\bstaff\b", tl))
        is_senior_staff = bool(re.search(r"\bsenior staff\b", tl))
        is_manager      = bool(re.search(r"\bmanager\b|\bmanagement\b", tl))
        is_sr_manager   = bool(re.search(r"\bsenior.*manager\b|\bsr\.?\s+manager\b", tl))
        is_director     = bool(re.search(r"\bdirector\b", tl))
        is_principal    = bool(re.search(r"\bprincipal\b", tl))
        is_distinguished= bool(re.search(r"\bdistinguished\b", tl))
        if jp == "early":
            return ("Entry", "stated_google")
        if jp == "mid":
            if is_senior_staff: return ("Senior Staff", "stated_google_enriched")
            if is_staff:        return ("Staff",        "stated_google_enriched")
            if is_sr_manager:   return ("Senior Manager","stated_google_enriched")
            if is_manager:      return ("Manager",       "stated_google_enriched")
            return ("Mid", "stated_google")
        if jp == "advanced":
            if is_distinguished:  return ("Distinguished",  "stated_google_enriched")
            if is_principal:      return ("Principal",      "stated_google_enriched")
            if is_senior_staff:   return ("Senior Staff",   "stated_google_enriched")
            if is_staff:          return ("Staff",          "stated_google_enriched")
            if is_director:
                return ("Senior Director","stated_google_enriched") \
                    if re.search(r"\bsenior director\b", tl) \
                    else ("Director", "stated_google_enriched")
            if is_sr_manager:     return ("Senior Manager", "stated_google_enriched")
            if is_manager:        return ("Manager",        "stated_google_enriched")
            if re.search(r"\bgroup\b.*(?:manager|product|lead)", tl):
                return ("Group Lead", "stated_google_enriched")
            return ("Advanced", "stated_google")

    if is_google and null_jp:
        if re.search(r"\bdistinguished\b",  tl): return ("Distinguished",  "title_keyword")
        if re.search(r"\bprincipal\b",      tl): return ("Principal",      "title_keyword")
        if re.search(r"\bsenior director\b",tl): return ("Senior Director","title_keyword")
        if re.search(r"\bdirector\b",       tl): return ("Director",       "title_keyword")
        if re.search(r"\bsenior staff\b",   tl): return ("Senior Staff",   "title_keyword")
        if re.search(r"\bstaff\b",          tl): return ("Staff",          "title_keyword")
        if re.search(r"\bsenior\b",         tl): return ("Senior",         "title_keyword")
        return ("Unspecified", "unavailable")

    # Amazon — numeric suffix, then prefix keywords
    if re.search(r"(?:^|[\s,\-])iii(?:[\s,\-]|$)", tl): return ("Senior",    "title_suffix_III")
    if re.search(r"(?:^|[\s,\-])ii(?:[\s,\-]|$)",  tl): return ("Mid",       "title_suffix_II")
    if re.search(r"(?:^|[\s,])i(?:,|\s|$)", tl):
        if not re.search(r"[a-z]i(?:,|\s|$)", tl):
            return ("Entry", "title_suffix_I")
    if re.search(r"^(?:sr\.?\s+|sr\s+)",                         tl): return ("Senior",    "title_prefix")
    if re.search(r"\bsenior\b",                                   tl): return ("Senior",    "title_prefix")
    if re.search(r"\bprincipal\b",                                tl): return ("Principal", "title_prefix")
    if re.search(r"\bstaff\b",                                    tl): return ("Staff",     "title_prefix")
    if re.search(r"\bjunior\b|\bjr\.?\b|\bassociate\b|\bentry\b", tl): return ("Entry",    "title_prefix")
    if re.search(r"\blead\b",                                     tl): return ("Lead",      "title_prefix")
    return ("Unspecified", "unavailable")


# ─────────────────────────────────────────────────────────────
# STEP 4 — TEAM EXTRACTOR
# ─────────────────────────────────────────────────────────────

TITLE_NOISE = re.compile(
    r"^(?:amazon|aws|amzl?|de|i{1,3}|ii|iii|"
    r"gtmo data|jwo data|scde|mods|auta)$",
    re.IGNORECASE
)

def extract_team(cleaned_title: str) -> str:
    parts = re.split(r"[,\-–]", cleaned_title, maxsplit=1)
    if len(parts) < 2:
        return ""
    team = parts[1].strip()
    if TITLE_NOISE.match(team) or re.match(r"^AMZ\d+", team, re.IGNORECASE):
        return ""
    return team


# ─────────────────────────────────────────────────────────────
# STEP 5 — DOMAIN TAGGER
# ─────────────────────────────────────────────────────────────

DOMAIN_RULES = [
    (r"fintech|finance|financial|tax|payroll|treasury|fp&a|gfs",    "Finance & FinTech"),
    (r"fba|fulfillment|shopbop|logistics|last mile|amzl|shiptech",  "Fulfillment & Logistics"),
    (r"alexa|echo|ring|blink|devices",                              "Devices & Alexa"),
    (r"aws|cloud|sagemaker|glue|redshift|kinesis|s3\b|bedrock",     "AWS Cloud"),
    (r"advertising|ads|sponsored|dsp\b|digi\b",                     "Advertising"),
    (r"prime video|video|luna|streaming",                           "Prime Video"),
    (r"seller|selling|marketplace|merchant",                        "Seller Services"),
    (r"payments|pay\b|lending",                                     "Payments"),
    (r"robotics|frontier ai|robot",                                 "Robotics"),
    (r"health|healthcare|pharma",                                   "Healthcare"),
    (r"music|books|publishing|kindle",                              "Content & Media"),
    (r"retail|stores|shopping|consumer",                            "Retail"),
    (r"security|compliance|fraud|trust|risk",                       "Security & Trust"),
]

def tag_domain(title: str) -> str:
    if not title:
        return "General/Other"
    t = title.lower()
    for pattern, domain in DOMAIN_RULES:
        if re.search(pattern, t, re.IGNORECASE):
            return domain
    return "General/Other"


# ─────────────────────────────────────────────────────────────
# STEP 6 — LOCATION PARSERS
# Amazon: ISO 3166-1 alpha-2 lookup ("US, WA, Bellevue")
# Google: city-first pattern match  ("Mountain View, CA, USA")
# ─────────────────────────────────────────────────────────────

AMAZON_COUNTRY_MAP = {
    "AE":"United Arab Emirates","AR":"Argentina",     "AT":"Austria",
    "AU":"Australia",           "BR":"Brazil",        "CA":"Canada",
    "CH":"Switzerland",         "CL":"Chile",         "CN":"China",
    "DE":"Germany",             "EG":"Egypt",         "ES":"Spain",
    "FR":"France",              "GB":"United Kingdom","HK":"Hong Kong",
    "IE":"Ireland",             "IL":"Israel",        "IN":"India",
    "IT":"Italy",               "JP":"Japan",         "KR":"South Korea",
    "LU":"Luxembourg",          "MX":"Mexico",        "MY":"Malaysia",
    "NL":"Netherlands",         "PH":"Philippines",   "PL":"Poland",
    "RO":"Romania",             "SA":"Saudi Arabia",  "SE":"Sweden",
    "SG":"Singapore",           "SK":"Slovakia",      "TH":"Thailand",
    "TW":"Taiwan",              "US":"United States", "VN":"Vietnam",
    "ZA":"South Africa",
}
CITY_NORMALISE = {"bangalore": "Bengaluru", "bombay": "Mumbai"}

def parse_amazon_location(raw: str) -> tuple:
    if not raw or str(raw).strip().lower() in ("null","none",""):
        return ("Unknown","Unknown")
    loc   = re.sub(r"\s*-\s*virtual\s*.*$","",str(raw).strip(),flags=re.IGNORECASE)
    parts = [p.strip() for p in loc.split(",") if p.strip()]
    if not parts:
        return ("Unknown","Unknown")
    country = AMAZON_COUNTRY_MAP.get(parts[0].upper(),"Other")
    city    = parts[-1].strip()
    if city.isdigit() and len(parts)>=2:
        city = parts[-2].strip()
    city = CITY_NORMALISE.get(city.lower(), city)
    return (city, country)

GOOGLE_COUNTRY_PATTERNS = [
    (r",\s*(?:CA|NY|WA|TX|VA|MA|GA|IL|CO|NC|OH|PA|AZ|FL|NJ|MN|OR|IN|WI|"
     r"MO|TN|OK|NE|SC|IA|NV|MD|DC|KY|MI|HI|UT)\s*,\s*USA$","United States"),
    (r",\s*USA$","United States"),(r"\bUSA\b","United States"),
    (r",\s*(?:Karnataka|Telangana|Maharashtra|Tamil Nadu|Haryana)\s*,\s*India$","India"),
    (r",\s*India$","India"),(r",\s*UK$","United Kingdom"),
    (r",\s*Australia$","Australia"),(r",\s*Poland$","Poland"),
    (r",\s*Germany$","Germany"),(r",\s*Switzerland$","Switzerland"),
    (r",\s*France$","France"),(r",\s*Japan$","Japan"),
    (r",\s*Taiwan$","Taiwan"),(r"Hsinchu County","Taiwan"),
    (r",\s*Israel$","Israel"),(r",\s*Brazil$","Brazil"),
    (r",\s*Ireland$","Ireland"),
    (r",\s*(?:ON|BC|QC|AB)\s*,\s*Canada$","Canada"),(r",\s*Canada$","Canada"),
    (r"^Singapore$|,\s*Singapore$","Singapore"),(r",\s*China$","China"),
    (r",\s*South Korea$","South Korea"),(r",\s*Netherlands$","Netherlands"),
    (r",\s*Sweden$","Sweden"),(r",\s*Denmark$","Denmark"),
    (r",\s*Finland$","Finland"),
    (r"Chon Buri.*Thailand$|Samut Prakan.*Thailand$|,\s*Thailand$","Thailand"),
    (r"Negeri Sembilan.*Malaysia$|Federal Territory.*Malaysia$|,\s*Malaysia$","Malaysia"),
    (r"Metro Manila|Taguig.*Philippines|,\s*Philippines$","Philippines"),
    (r"CDMX.*Mexico$|,\s*Mexico$","Mexico"),
    (r"Metropolitan City.*Italy$|,\s*Italy$","Italy"),
    (r",\s*Spain$","Spain"),(r",\s*Austria$","Austria"),
    (r",\s*Romania$","Romania"),(r",\s*Belgium$","Belgium"),
    (r",\s*Hungary$","Hungary"),
    (r"Vilnius City.*Lithuania$|,\s*Lithuania$","Lithuania"),
    (r"İstanbul.*Türkiye$|,\s*Türkiye$","Turkey"),
    (r",\s*South Africa$","South Africa"),(r",\s*New Zealand$","New Zealand"),
    (r",\s*Kenya$","Kenya"),(r",\s*Colombia$","Colombia"),
    (r",\s*Argentina$","Argentina"),(r",\s*Chile$","Chile"),
    (r"^Hong Kong$","Hong Kong"),(r",\s*Norway$","Norway"),
    (r",\s*Luxembourg$","Luxembourg"),(r",\s*Indonesia$","Indonesia"),
    (r",\s*Czechia$","Czech Republic"),
]

def parse_google_location(raw: str) -> tuple:
    if not raw or str(raw).strip() in ("","null","none"):
        return ("Unknown","Unknown")
    loc     = str(raw).strip()
    country = "Other"
    for pattern, cntry in GOOGLE_COUNTRY_PATTERNS:
        if re.search(pattern, loc, re.IGNORECASE):
            country = cntry
            break
    raw_city = loc.split(",")[0].strip()
    city     = re.sub(r"\s+[A-Z]{2,3}$","",raw_city).strip() or raw_city
    return (city, country)


# ─────────────────────────────────────────────────────────────
# UDF REGISTRATION
# ─────────────────────────────────────────────────────────────

# COMMAND ----------

clean_title_udf     = F.udf(clean_title,        StringType())
classify_family_udf = F.udf(classify_job_family, StringType())
tag_domain_udf      = F.udf(tag_domain,          StringType())
extract_team_udf    = F.udf(extract_team,        StringType())

SENIORITY_SCHEMA = StructType([
    StructField("seniority_level",  StringType(), False),
    StructField("seniority_source", StringType(), False),
])

@F.pandas_udf(SENIORITY_SCHEMA)
def extract_seniority_udf(
    titles: pd.Series, positions: pd.Series, orgs: pd.Series
) -> pd.DataFrame:
    results = [extract_seniority(t,p,o) for t,p,o in zip(titles,positions,orgs)]
    return pd.DataFrame(results, columns=["seniority_level","seniority_source"])

LOCATION_SCHEMA = StructType([
    StructField("city",    StringType(), False),
    StructField("country", StringType(), False),
])

@F.pandas_udf(LOCATION_SCHEMA)
def parse_location_udf(raw_locations: pd.Series, org_keys: pd.Series) -> pd.DataFrame:
    results = [
        parse_amazon_location(loc) if "amazon" in (org or "").lower()
        else parse_google_location(loc)
        for loc, org in zip(raw_locations, org_keys)
    ]
    return pd.DataFrame(results, columns=["city","country"])


# ─────────────────────────────────────────────────────────────
# NORMALIZE FUNCTION
# ─────────────────────────────────────────────────────────────

def normalize_titles(df):
    return (
        df
        .withColumn("org_key",     F.lower(F.trim(F.col("organization"))))
        .withColumn("title_clean", clean_title_udf(F.col("title")))
        .withColumn("job_family",  classify_family_udf(F.col("title_clean")))
        .withColumn("_seniority",  extract_seniority_udf(
            F.col("title_clean"),
            F.coalesce(F.col("job_position"), F.lit("")),
            F.col("org_key"),
        ))
        .withColumn("seniority",        F.col("_seniority.seniority_level"))
        .withColumn("seniority_source", F.col("_seniority.seniority_source"))
        .drop("_seniority")
        .withColumn("team_name",   extract_team_udf(F.col("title_clean")))
        .withColumn("domain",      tag_domain_udf(F.col("title_clean")))
        .withColumn("is_intern",   F.col("seniority") == "Intern")
        .withColumn("_location",   parse_location_udf(F.col("job_location"), F.col("org_key")))
        .withColumn("city",              F.col("_location.city"))
        .withColumn("country",           F.col("_location.country"))
        .withColumn("job_location_clean", F.col("_location.city"))
        .drop("_location")
    )


# ─────────────────────────────────────────────────────────────
# NULL CHECKS
# ─────────────────────────────────────────────────────────────

LITERAL_NULL_COLS = ["job_position", "job_description", "job_location"]

def build_quarantine_flags(df):
    df_flagged = df
    for col_name in LITERAL_NULL_COLS:
        if col_name in df.columns:
            df_flagged = df_flagged.withColumn(
                col_name,
                F.when(F.lower(F.trim(F.col(col_name))).isin("null","none",""),None)
                 .otherwise(F.col(col_name))
            )
    reason_expr = (
        F.when(F.col("doc_id").isNull(),                           F.lit("null:doc_id"))
        .when(F.col("job_description").isNull(),                   F.lit("null:job_description"))
        .when(F.col("organization").isNull(),                      F.lit("null:organization"))
        .when(F.col("title").isNull(),                             F.lit("null:title"))
        .when(F.col("file_dt").isNull(),                           F.lit("null:file_dt"))
        .when(F.length(F.trim(F.col("job_description"))) < 50,    F.lit("too_short:job_description"))
        .otherwise(None)
    )
    df_flagged    = df_flagged.withColumn("quarantine_reason", reason_expr)
    df_clean      = df_flagged.filter(F.col("quarantine_reason").isNull()).drop("quarantine_reason")
    df_quarantine = df_flagged.filter(F.col("quarantine_reason").isNotNull()) \
                               .withColumn("quarantined_at", F.current_timestamp())
    return df_clean, df_quarantine


def handle_soft_nulls(df):
    flags_expr = F.array_remove(F.array(
        F.when(F.col("job_location").isNull(),
               F.lit("missing:job_location")).otherwise(F.lit(None).cast(StringType())),
        F.when(F.col("source_url").isNull(),
               F.lit("missing:source_url")).otherwise(F.lit(None).cast(StringType())),
        F.when(F.col("job_id").isNull(),
               F.lit("missing:job_id")).otherwise(F.lit(None).cast(StringType())),
        F.when(F.col("source_type").isNull(),
               F.lit("missing:source_type")).otherwise(F.lit(None).cast(StringType())),
        F.when(F.col("notional_job_posted_dt").isNull(),
               F.lit("warn:no_posted_date_using_file_dt")).otherwise(F.lit(None).cast(StringType())),
    ), None)
    return (
        df
        .withColumn("data_quality_flags", flags_expr)
        .withColumn("job_location",  F.coalesce(F.col("job_location"),  F.lit("Location not specified")))
        .withColumn("source_url",    F.coalesce(F.col("source_url"),    F.lit("No URL")))
        .withColumn("job_id",        F.coalesce(F.col("job_id"),        F.lit("Unknown")))
        .withColumn("source_type",   F.coalesce(F.col("source_type"),   F.lit("web")))
        .withColumn("trend_dt",      F.coalesce(
            F.to_date(F.col("notional_job_posted_dt")),
            F.to_date(F.col("file_dt"))
        ))
    )


def validate_derived_columns(df_normalized):
    DERIVED_HARD = ["title_clean", "job_family", "seniority", "seniority_source"]
    reason_expr  = F.lit(None).cast(StringType())
    for col_name in DERIVED_HARD:
        reason_expr = F.when(
            F.col(col_name).isNull(), F.lit(f"udf_failed:{col_name}")
        ).otherwise(reason_expr)
    df_checked    = df_normalized.withColumn("quarantine_reason", reason_expr)
    df_clean      = df_checked.filter(F.col("quarantine_reason").isNull()).drop("quarantine_reason")
    df_quarantine = df_checked.filter(F.col("quarantine_reason").isNotNull()) \
                               .withColumn("quarantined_at", F.current_timestamp())
    return df_clean, df_quarantine


# ─────────────────────────────────────────────────────────────
# WATERMARK-GATED PIPELINE
# ─────────────────────────────────────────────────────────────

# COMMAND ----------


def get_readyness_parameters(wm: object) -> tuple:
    try:
        availability = wm.check_availability()
        orgs_ready   = [o for o, i in availability.items() if i["ready"]]
        orgs_waiting = [o for o, i in availability.items() if not i["ready"]]
        orgs_failed  = [o for o, i in availability.items()
                        if not i["ready"] and i.get("last_status") == "failed"]
        
        return availability, orgs_ready, orgs_waiting, orgs_failed
    except Exception as e:
        print(e)
        return {}, [], [], []


def check_availability_gate(wm: object, availability: dict, orgs_ready: list, orgs_waiting: list, orgs_failed: list) -> bool:
    if REQUIRE_ALL_ORGS:
        if orgs_failed:
            try:
                reason = f"blocked by failed orgs: {orgs_failed}"
                for org in availability:
                    wm.mark_skipped(org, reason)
                print(f"\nPipeline blocked — {reason}")
                return False
            except Exception as e:
                print(e)
                return False

        if orgs_waiting and not orgs_ready:
            try:
                reason = f"waiting for new file_dt from: {orgs_waiting}"
                for org in availability:
                    wm.mark_skipped(org, reason)
                print(f"\nPipeline skipped — {reason}")
                return False
            except Exception as e:
                print(e)
                return False

    if not orgs_ready:
        print("No orgs have new data. Nothing to process.")
        return False

    return True


def process_incrementally(wm, availability, orgs_ready):
    """Per-org incremental processing."""
    for org in orgs_ready:
        last_wm    = availability[org]["last_watermark"]
        max_new_dt = availability[org]["max_new_dt"]

        print(f"\n{'─' * 55}")
        print(f"  [{org.upper()}]  window: ({last_wm}, {max_new_dt}]")
        print(f"{'─' * 55}")

        try:
            # Load ONLY new partitions — never reprocess old data
            df_raw = (
                spark.table(RAW_TABLE)
                .select(
                    "doc_id","source_type","source_url","organization","job_id",
                    "title","job_location","job_position","job_description",
                    "ingested_at","file_dt","notional_job_posted_dt"
                )
                .filter(F.lower(F.col("organization")) == org)
                .filter(F.col("file_dt") >  F.lit(last_wm))
                .filter(F.col("file_dt") <= F.lit(max_new_dt))
                .dropDuplicates(["doc_id"])
            )

            raw_count = df_raw.count()
            print(f"  Raw records in window: {raw_count}")
            if raw_count == 0:
                wm.mark_skipped(org, f"0 rows in ({last_wm}, {max_new_dt}]")
                continue

            # Stage 1: Hard null quarantine
            df_clean, df_q_hard = build_quarantine_flags(df_raw)
            if df_q_hard.count() > 0:
                df_q_hard.write.format("delta").mode("append") \
                    .option("mergeSchema","true").saveAsTable(QUARANTINE_TABLE)
                print(f"  Hard quarantined: {df_q_hard.count()}")

            # Stage 2: Soft null imputation
            df_soft = handle_soft_nulls(df_clean)

            # Stage 3: Normalize
            df_normalized = normalize_titles(df_soft)

            # Stage 4: Derived column check
            df_final, df_q_derived = validate_derived_columns(df_normalized)
            if df_q_derived.count() > 0:
                df_q_derived.write.format("delta").mode("append") \
                    .option("mergeSchema","true").saveAsTable(QUARANTINE_TABLE)
                print(f"  Derived quarantined: {df_q_derived.count()}")

            final_count = df_final.count()
            print(f"  Clean records: {final_count}")

            # Stage 5: MERGE into clean_openings — idempotent on doc_id
            # Table already created by bootstrap_watermark.py with CDF enabled
            df_final.createOrReplaceTempView(f"clean_source_{org}")
            spark.sql(f"""
                MERGE INTO {CLEAN_TABLE} AS T
                USING clean_source_{org} AS S
                ON T.doc_id = S.doc_id
                WHEN MATCHED     THEN UPDATE SET *
                WHEN NOT MATCHED THEN INSERT *
            """)
            print(f"  Merged {final_count} rows into {CLEAN_TABLE}")

            print(f"Inside the try block before logging success")

            # Advance watermark — ONLY after full success
            wm.mark_success(org, max_new_dt, final_count, stage="normalize")

        except Exception as e:
            print(f"Inside the error block")
            wm.mark_failed(org, str(e), stage="normalize")
            return False

    return True


def main() -> None:
    wm = WatermarkManager(spark)
    availability, orgs_ready, orgs_waiting, orgs_failed = get_readyness_parameters(wm)
    available_gate = check_availability_gate(wm, availability, orgs_ready, orgs_waiting, orgs_failed)
    
    if available_gate:
        if not process_incrementally(wm, availability, orgs_ready):
            print("Pipeline failed in cleaning data")
            
        wm.status()
        return None
    else:
        print("There is a problem with availability")
        return None


if __name__ == '__main__':
    main()
