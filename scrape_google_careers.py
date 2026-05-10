# ══════════════════════════════════════════════════════════════════════════════
# 01_scrape_google_careers.py
#
# Watermark interactions (3 only — scraping logic unchanged):
#   1. register_org("google")      — idempotent, cached after first call
#   2. already_scraped guard       — reads raw_openings (NOT watermark table)
#   3. mark_skipped / mark_no_data / mark_failed depending on outcome
#
# Does NOT advance last_processed_dt — that belongs to 03_data_cleaning.
# ══════════════════════════════════════════════════════════════════════════════
import os
from pipeline_watermark import WatermarkManager
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DateType
from pyspark.sql import Row, DataFrame, functions as F, Window
import requests, uuid, re
from datetime import datetime, timezone, date
from bs4 import BeautifulSoup
from properties import web_browser_agent,RAW_TABLE, google_org_id,DOCUMENT_SCHEMA,ingestion_stage
# from logger_config import get_logger

# logger = get_logger("scrape_google")

JOB_ID_PATTERN = r"jobs/results/(\d+)"  #check if we can move this to properties
today=date.today()
page_url=""
individual_job_url="https://www.google.com/about/careers/applications/"
# ─────────────────────────────────────────────────────────────
# WATERMARK — register + skip-guard
# ─────────────────────────────────────────────────────────────
def check_if_rerun(wm_obj,org_id)-> bool:
    try:
        already_scraped = (
        spark.table(RAW_TABLE)
            .filter(
                (F.lower(F.col("scrape_org_key"))==F.lit(org_id)) &
                (F.col("file_dt") == F.lit(today))
            )
            .limit(1)
            .count()
        ) > 0

        if already_scraped:
            wm_obj.mark_skipped(org_id, f"file_dt={today} already present in raw_openings")

    except Exception as e:
        # Bare except silently swallowed Unity Catalog hiccups and let the
        # scraper proceed as if no data existed — that's how a partial earlier
        # run could end up with a second copy of every row. Log and treat the
        # check as inconclusive (safer to skip than to risk duplicates, but
        # keeping the original behaviour of attempting the scrape since the
        # MERGE itself is now duplicate-safe).
        print(f"[google] check_if_rerun error: {type(e).__name__}: {e}")
        already_scraped = False
    return already_scraped


# ─────────────────────────────────────────────────────────────
# SCRAPING HELPERS
# ─────────────────────────────────────────────────────────────

def get_job_title(soup):
    try:    return soup.find("h2", {"class": "p1N2lc"}).get_text(strip=True)
    except: return ""

def get_job_organization(soup):
    try:    return soup.find("span", {"class": "RP7SMd"}).find("span").get_text(strip=True)
    except: return ""

def get_job_location(soup):
    try:    return soup.find("span", {"class": "pwO9Dc vo5qdf"}).find("span").get_text(strip=True)
    except: return ""

def get_job_position(soup):
    try:    return soup.find("span", {"class": "wVSTAb"}).get_text(strip=True)
    except: return ""

def get_job_qualification(soup):
    try:    return soup.find("div", {"class": "KwJkGe"}).get_text(separator="\n", strip=True)
    except: return ""

def get_about_job(soup):
    try:    return soup.find("div", {"class": "aG5W3"}).get_text(separator="\n", strip=True)
    except: return ""

def get_job_responsibility(soup):
    try:    return soup.find("div", {"class": "BDNOWe"}).get_text(separator="\n", strip=True)
    except: return ""

def scrape_job(link: str, job_id: str) -> Row:
    resp = requests.get(link, headers={"User-Agent": web_browser_agent}, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")
    return Row(
        doc_id                = str(uuid.uuid4()),
        source_type           = "web",
        source_url            = link,
        organization          = get_job_organization(soup),
        job_id                = job_id,
        title                 = get_job_title(soup),
        job_location          = get_job_location(soup),
        job_position          = get_job_position(soup),
        job_description       = (f"{get_about_job(soup)}\n"
                                 f"{get_job_qualification(soup)}\n"
                                 f"{get_job_responsibility(soup)}"),
        ingested_at           = datetime.now(timezone.utc),
        file_dt               = today,
        notional_job_posted_dt= today,
        scrape_org_key        = google_org_id
    )


# ─────────────────────────────────────────────────────────────
# SCRAPE
# ─────────────────────────────────────────────────────────────

def start_scraping() -> DataFrame:
    trades = ["data", "ai", "agentic", "software"]
    rows   = []
    seen_ids: set = set()  # persist across ALL trades to prevent cross-keyword duplicates

    for trade in trades:
        total_jobs   = 0
        crawled_jobs = 0
        for page in range(1, 11):
            url  = (f"https://www.google.com/about/careers/applications/"
                    f"jobs/results?q={trade}&page={page}")
            resp = requests.get(url, headers={"User-Agent": web_browser_agent}, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "html.parser")

            if page == 1:
                match = re.search(
                    r"of\s+(\d+)",
                    soup.find("div", attrs={"jsname": "uEp2ad"}).get_text()
                )
                if match:
                    total_jobs = int(match.group(1))

            for link_tag in soup.find_all("a", attrs={"class": "WpHeLc VfPpkd-mRLv6 VfPpkd-RLmnJb"}):
                href  = link_tag.get("href", "")
                match = re.search(JOB_ID_PATTERN, href)
                if not match:
                    continue
                job_id   = match.group(1)
                if job_id in seen_ids:
                    continue
                seen_ids.add(job_id)
                full_url = f"{individual_job_url}{href}"
                try:
                    rows.append(scrape_job(full_url, job_id))
                except Exception as e:
                    # logger.warning("Skipped job %s: %s", job_id, e)
                    print(f"Skipped job {job_id}: {e}")

            crawled_jobs += len(rows)
            if total_jobs and crawled_jobs >= total_jobs:
                break
    # Convert list of Rows to a Spark DataFrame
    return spark.createDataFrame(rows, schema=DOCUMENT_SCHEMA)
    

def _dedupe_for_merge(df: DataFrame) -> DataFrame:
    """
    Collapse source-side duplicates before MERGE.

    Why this is necessary: Delta MERGE evaluates each source row against the
    target snapshot taken at the start of the operation. With WHEN NOT MATCHED
    only, two source rows that share (job_id, organization) but find no target
    match BOTH get classified as "not matched" and BOTH get inserted. The
    Google scraper sweeps four trade keywords across ten pages each, so any
    job that matches multiple keywords arrives in the source N times.

    Dedup rule:
      partition by (job_id, lower(organization))     -- canonical key
      order by ingested_at DESC, length(description) -- prefer the freshest,
                                                       richest record
      keep the top row.

    Also drops rows with NULL job_id (the Google JOB_ID_PATTERN regex already
    skips these, but the filter is cheap insurance for future scraper changes).
    """
    w = (Window
         .partitionBy(F.col("job_id"),
                      F.lower(F.col("organization")))
         .orderBy(F.col("ingested_at").desc(),
                  F.length(F.col("job_description")).desc()))
    return (
        df
        .filter(F.col("job_id").isNotNull() & (F.col("job_id") != ""))
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )


def merge_result(rowsDf: DataFrame) -> bool:
    try:
        # 1. Source-side dedup. Without this the MERGE inserts duplicates.
        before = rowsDf.count()
        rowsDf = _dedupe_for_merge(rowsDf)
        after  = rowsDf.count()
        if before != after:
            print(f"[google] dedup collapsed {before} → {after} source rows")

        rowsDf.createOrReplaceTempView("google_source")

        # 2. MERGE on lower(organization) so case differences in the parsed
        #    org name (Google / google / YouTube vs youtube) don't create
        #    parallel keys. The WHEN MATCHED THEN UPDATE clause turns this
        #    into a true upsert and engages Delta's cardinality safety net:
        #    if dedup ever fails to collapse all source duplicates, the
        #    MERGE will now error loudly instead of silently double-inserting.
        spark.sql(f"""
            MERGE INTO {RAW_TABLE} AS T
            USING google_source AS S
            ON  T.job_id              = S.job_id
            AND lower(T.organization) = lower(S.organization)
            WHEN MATCHED THEN UPDATE SET
                ingested_at = S.ingested_at
            WHEN NOT MATCHED THEN INSERT (
                doc_id, source_type, source_url, organization, job_id,
                title, job_location, job_position, job_description,
                ingested_at, file_dt, notional_job_posted_dt, scrape_org_key
            ) VALUES (
                S.doc_id, S.source_type, S.source_url,
                lower(S.organization), S.job_id,
                S.title, S.job_location, S.job_position, S.job_description,
                S.ingested_at, S.file_dt, S.file_dt, S.scrape_org_key
            )
        """)
        print(f"Google MERGE complete -> {RAW_TABLE}")
        return True
    except Exception as e:
        print(f"[google] MERGE failed: {type(e).__name__}: {e}")
        return False


def main()->None:
    wm    = WatermarkManager(spark)
    wm.register_org(google_org_id, stage=ingestion_stage, seed_dt=today)
    search_result = check_if_rerun(wm, google_org_id)
    if not search_result:
        try:
            scrapped_df = start_scraping()    
            if scrapped_df.isEmpty():
                # logger.warning("Zero rows scraped — nothing to merge for %s", today)
                print(f"Zero rows scraped — nothing to merge for {today}")
                wm.mark_no_data(google_org_id, stage=ingestion_stage) 
            else:
                print(f"Google scrape complete -> {scrapped_df.count()} rows scraped")
                success = merge_result(scrapped_df)
                if success:
                    print(f"Google MERGE complete -> {RAW_TABLE}")
                    # wm.mark_success(google_org_id, today,0)
                else:
                    wm.mark_failed(google_org_id, "MERGE into raw_openings failed")
        except Exception as e:
            print(f"Error scraping Google: {e}")
            wm.mark_failed(google_org_id, f"Error scraping Google: {e}")
            raise

    else:
        print(f"Google scrape skipped — {today} already loaded")
    
    return None

if __name__ == "__main__":
    main()