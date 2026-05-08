
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DateType






# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

catalog     = "bootcamp_students"
schema_name = "zachy_balaji_kottana05"

google_org_id="google"
amazon_org_id="amazon"
openai_org_id="openai"
ingestion_stage="normalize"
chunk_stage="chunk"
vector_stage="vectorize"



openai_careers_url="https://openai.com/careers/search/"




RAW_TABLE        = f"{catalog}.{schema_name}.raw_openings"
CLEAN_TABLE      = f"{catalog}.{schema_name}.clean_openings"
WATERMARK_TABLE  = f"{catalog}.{schema_name}.pipeline_watermarks"
HISTORY_TABLE    = f"{catalog}.{schema_name}.pipeline_watermark_history"
QUARANTINE_TABLE = f"{catalog}.{schema_name}.quarantine_openings"
QUARANTINE_SCRAPE_URLS = f"{catalog}.{schema_name}.scraped_error_logs"

CHUNKS_TABLE = f"{catalog}.{schema_name}.job_sentence_chunks"

#chunking parameters
WINDOW_SIZE      = 2    # sentences each side of focal sentence
MIN_SENTENCE_LEN = 20   # discard fragments shorter than this

#vector search parameters
VS_INDEX     = f"{catalog}.{schema_name}.job_chunks_index"  
EMBED_MODEL  = "databricks-gte-large-en"         # Databricks-hosted embedding model      
# Databricks Vector Search config
#VS_ENDPOINT  = "job_market_vs"
VS_ENDPOINT  = "zachy_vs"                                    # endpoint name — shared across workspace

EMBED_COLUMN = "sentence"                                         # focal sentence — what gets embedded

web_browser_agent = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/26.2 Safari/605.1.15"
)

DOCUMENT_SCHEMA = StructType([
    StructField("doc_id",                StringType(),    False),
    StructField("source_type",           StringType(),    False),
    StructField("source_url",            StringType(),    False),
    StructField("organization",          StringType(),    True),
    StructField("job_id",                StringType(),    True),
    StructField("title",                 StringType(),    True),
    StructField("job_location",          StringType(),    True),
    StructField("job_position",          StringType(),    True),
    StructField("job_description",       StringType(),    True),
    StructField("ingested_at",           TimestampType(), False),
    StructField("file_dt",               DateType(),      False),
    StructField("notional_job_posted_dt",DateType(),      False),
    StructField("scrape_org_key",        StringType(),    False)
])




# ── company registry ──────────────────────────────────────────────────────────

COMPANIES = {
    "ashby": {
        "openai","notion","figma","anthropic",
        "vercel","mistralai","perplexity","cohere",
        "airbyte","airtable","anomalo","anyscale",
        "clerk","deel","deepl","elevenlabs","gamma",
        "hightouch","linear","loom","mercury","modal",
        "neon","railway","ramp","render","retool",
        "runway","supabase","synthesia","snowflake",
        "langchain",
    },
}

# COMPANIES = {

#     # ── Ashby ─────────────────────────────────────────────────────────────────
#     "openai":       {"ats": "ashby",      "slug": "openai"},
#     "notion":       {"ats": "ashby",      "slug": "notion"},
#     #"figma":        {"ats": "ashby",      "slug": "figma"},
#     "anthropic":    {"ats": "ashby",      "slug": "anthropic"},
#     #"vercel":       {"ats": "ashby",      "slug": "vercel"},
#     #"mistralai":    {"ats": "ashby",      "slug": "mistralai"},
#     "perplexity":   {"ats": "ashby",      "slug": "perplexity"},
#     "cohere":       {"ats": "ashby",      "slug": "cohere"},
#     "airbyte":      {"ats": "ashby",      "slug": "airbyte"},
#     "airtable":     {"ats": "ashby",      "slug": "airtable"},
    
#     "anomalo":      {"ats": "ashby",      "slug": "anomalo"},
#     "anyscale":     {"ats": "ashby",      "slug": "anyscale"},
#     "clerk":        {"ats": "ashby",      "slug": "clerk"},
#     "deel":         {"ats": "ashby",      "slug": "deel"},
#     "deepl":        {"ats": "ashby",      "slug": "deepl"},
#     "elevenlabs":   {"ats": "ashby",      "slug": "elevenlabs"},
#     "gamma":        {"ats": "ashby",      "slug": "gamma"},
#     "hightouch":    {"ats": "ashby",      "slug": "hightouch"},
#     "linear":       {"ats": "ashby",      "slug": "linear"},


#     #"loom":         {"ats": "ashby",      "slug": "loom"},
#     "mercury":      {"ats": "ashby",      "slug": "mercury"},
#     "modal":        {"ats": "ashby",      "slug": "modal"},
#     "neon":         {"ats": "ashby",      "slug": "neon"},
#     "railway":      {"ats": "ashby",      "slug": "railway"},
#     "ramp":         {"ats": "ashby",      "slug": "ramp"},
#     "render":       {"ats": "ashby",      "slug": "render"},
#     "retool":       {"ats": "ashby",      "slug": "retool"},
#     "runway":       {"ats": "ashby",      "slug": "runway"},
#     "supabase":     {"ats": "ashby",      "slug": "supabase"},
#     "synthesia":    {"ats": "ashby",      "slug": "synthesia"},
#     "snowflake":    {"ats": "ashby",      "slug": "snowflake"},
#     "langchain":    {"ats": "ashby",      "slug": "langchain"},
    
      

#     # # ── Greenhouse ────────────────────────────────────────────────────────────
#     # "airbnb":       {"ats": "greenhouse", "slug": "airbnb"},
#     # "coinbase":     {"ats": "greenhouse", "slug": "coinbase"},
#     # "dropbox":      {"ats": "greenhouse", "slug": "dropbox"},
#     # "reddit":       {"ats": "greenhouse", "slug": "reddit"},
#     # "robinhood":    {"ats": "greenhouse", "slug": "robinhood"},
#     # "stripe":       {"ats": "greenhouse", "slug": "stripe"},
#     # "doordash":     {"ats": "greenhouse", "slug": "doordash"},
#     # "lyft":         {"ats": "greenhouse", "slug": "lyft"},
#     # "twilio":       {"ats": "greenhouse", "slug": "twilio"},
#     # "snowflake":    {"ats": "greenhouse", "slug": "snowflake"},
#     # "databricks":   {"ats": "greenhouse", "slug": "databricks"},
#     # "palantir":     {"ats": "greenhouse", "slug": "palantir"},
#     # "confluent":    {"ats": "greenhouse", "slug": "confluent"},
#     # "dbtlabs":      {"ats": "greenhouse", "slug": "dbtlabs"},

#     # # ── Lever ─────────────────────────────────────────────────────────────────
#     # "spotify":      {"ats": "lever",      "slug": "spotify"},
#     # "atlassian":    {"ats": "lever",      "slug": "atlassian"},
#     # "scaleai":      {"ats": "lever",      "slug": "scaleai"},
#     # "anyscale":     {"ats": "lever",      "slug": "anyscale"},
#     # "togetherai":   {"ats": "lever",      "slug": "togetherai"},

#     # # ── Eightfold ─────────────────────────────────────────────────────────────
#     # "netflix":      {"ats": "eightfold",  "url": "https://explore.jobs.netflix.net/careers/search",  "domain": "netflix.com"},
#     # "cisco":        {"ats": "eightfold",  "url": "https://jobs.cisco.com/jobs/SearchJobs",           "domain": "cisco.com"},

#     # # ── Custom REST ───────────────────────────────────────────────────────────
#     # "amazon":       {"ats": "amazon"},
#     # "apple":        {"ats": "apple"},
#     # "microsoft":    {"ats": "microsoft"},

#     # # ── Avature (HTML) ────────────────────────────────────────────────────────
#     # "bloomberg":    {"ats": "avature",    "url": "https://bloomberg.avature.net/careers/SearchJobs"},
# }

