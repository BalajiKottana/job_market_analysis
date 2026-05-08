# Databricks notebook source
# ══════════════════════════════════════════════════════════════════════════════
# 02_scrape_amazon_careers.py
#
# Watermark interactions (same pattern as 01_scrape_google_careers.py):
#   1. register_org("amazon")      — idempotent, cached after first call
#   2. already_scraped guard       — reads raw_openings directly
#   3. mark_skipped / mark_no_data / mark_failed depending on outcome
#
# Does NOT advance last_processed_dt — that belongs to 03_data_cleaning.
# ══════════════════════════════════════════════════════════════════════════════

# COMMAND ----------

import os
from pipeline_watermark import WatermarkManager
from pyspark.sql import DataFrame, functions as F
from datetime import datetime, date
from bs4 import BeautifulSoup
import re, requests, time, uuid
from properties import RAW_TABLE,  amazon_org_id,ingestion_stage


today=date.today()
URL          = "https://www.amazon.jobs/en/search.json"
BASE_URL     = "https://www.amazon.jobs"
ORGANIZATION = "amazon"
SOURCE_TYPE  = "web"

TRENDS = [
    "data engineer", "data analyst", "data scientist", "data architect",
    "agent", "ai", "machine learning",
]
OFFSETS    = [0]
RATE_LIMIT = 100
HEADERS    = {
    "User-Agent":      "Mozilla/5.0",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept":          "application/json, text/javascript, */*",
    "Referer":         "https://www.amazon.jobs/",
}
FIELDS = {"id_icims", "job_category", "job_family", "job_path", "location", "posted_date"}

# ─────────────────────────────────────────────────────────────
# SCRAPING HELPERS
# ─────────────────────────────────────────────────────────────

# COMMAND ----------

def parse_job(job: dict) -> dict:
    parsed = {k: job.get(k) for k in FIELDS}
    try:
        parsed["posted_date"] = datetime.strptime(
            parsed["posted_date"], "%B %d, %Y"
        ).date()
    except (ValueError, TypeError):
        parsed["posted_date"] = None
    return parsed


def fetch_jobs(trend: str, offset: int, retries: int = 3,
               backoff: float = 2.0) -> list:
    params = {
        "base_query":   trend,
        "offset":       offset,
        "result_limit": RATE_LIMIT,
        "sort":         "relevant",
        "loc_query":    "",
        "country":      "",
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(URL, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            jobs = data.get("jobs") if isinstance(data, dict) else None
            return [parse_job(j) for j in jobs] if isinstance(jobs, list) else []
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else None
            if status == 429:
                time.sleep(backoff ** attempt * 5)
            elif isinstance(status, int) and 400 <= status < 500:
                return []
        except Exception:
            pass
        if attempt < retries:
            time.sleep(backoff ** attempt)
    return []


def fetch_job_description(job_path: str, retries: int = 3,
                          backoff: float = 2.0) -> dict:
    url    = f"{BASE_URL}{job_path}"
    result = {}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0",
                         "Accept-Encoding": "gzip, deflate, br"},
                timeout=30, stream=True,
            )
            resp.raise_for_status()
            soup     = BeautifulSoup(resp.text, "html.parser")
            sections = soup.select("#job-detail-body .section")
            try:    result["job_title"] = soup.find("h1", class_="title").get_text(strip=True)
            except: result["job_title"] = ""
            try:    result["job_category"] = soup.select_one(".association.job-category-icon a").get_text(strip=True)
            except: result["job_category"] = ""
            result["job_description"] = "\n".join(
                s.get_text(strip=True, separator="\n") for s in sections
            )
            return result
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else None
            if status == 429:
                time.sleep(backoff ** attempt * 5)
            elif isinstance(status, int) and 400 <= status < 500:
                return {}
        except Exception:
            return {}
        if attempt < retries:
            time.sleep(backoff ** attempt)
    return {}


def scrape_all_jobs() -> DataFrame:
    all_jobs: list = []
    for trend in TRENDS:
        seen_ids: set = set()
        for offset in OFFSETS:
            jobs = fetch_jobs(trend, offset)
            if not jobs:
                break
            for job in jobs:
                job_id = job.get("id_icims")
                if job_id and str(job_id) in seen_ids:
                    continue
                if job_id:
                    seen_ids.add(str(job_id))

                job_path = job.get("job_path")
                if job_path:
                    desc = fetch_job_description(job_path)
                    job.update({
                        "job_url":         f"{BASE_URL}{job_path}",
                        "job_title":       desc.get("job_title", ""),
                        "job_category":    desc.get("job_category", ""),
                        "job_description": desc.get("job_description", ""),

                    })
                    time.sleep(0.5)
                else:
                    job.update({"job_url": "", "job_title": "",
                                "job_category": "", "job_description": ""})

                job.update({
                    "organization": ORGANIZATION,
                    "source_type":  SOURCE_TYPE,
                    "file_dt":      today,
                    "ingested_at":  datetime.now(),
                    "doc_id":       str(uuid.uuid4()),
                    "scrape_org_key": amazon_org_id
                })
                all_jobs.append(job)
            time.sleep(0.5)

    # Convert list of dicts to a Spark DataFrame
    return spark.createDataFrame(all_jobs)


def check_if_rerun(wm_obj, org_id) -> bool:
    try:
        already_scraped = (
        spark.table(RAW_TABLE)
            .filter(
                (F.lower(F.col("scrape_org_key")) == F.lit(org_id)) &
                (F.col("file_dt") == F.lit(today))
            )
            .limit(1)
            .count()
        ) > 0
        
        if already_scraped:
            wm_obj.mark_skipped(org_id, f"file_dt={today} already present in raw_openings")
             
    except:
        print(f"Error checking for already scraped data")
        already_scraped = False
    return already_scraped


def merge_result(rowsDf: DataFrame) -> bool:
    try:
        rowsDf.createOrReplaceTempView("amazon_source")
        spark.sql(f"""
        MERGE INTO {RAW_TABLE} AS T
        USING amazon_source AS S
        ON  T.job_id       = S.id_icims
        AND T.organization = S.organization
        WHEN NOT MATCHED THEN INSERT (
            doc_id, source_type, source_url, organization, job_id,
            title, job_location, job_description,
            ingested_at, file_dt, notional_job_posted_dt,scrape_org_key
        ) VALUES (
            S.doc_id, S.source_type, S.job_url, S.organization, S.id_icims,
            S.job_title, S.location, S.job_description,
            S.ingested_at, S.file_dt, S.posted_date,S.scrape_org_key
        )
    """)
        print(f"Amazon MERGE complete → {RAW_TABLE}")
        return True
    except Exception as e:
        print(f"MERGE failed: {e}")
        return False


def main() -> None:
    wm    = WatermarkManager(spark)
    wm.register_org(amazon_org_id, stage=ingestion_stage, seed_dt=today)
    search_result = check_if_rerun(wm, amazon_org_id)
    if not search_result:
        try:
            scrapped_df = scrape_all_jobs()
            if scrapped_df.isEmpty():
                print(f"Zero rows scraped — nothing to merge for {today}")
                wm.mark_no_data(amazon_org_id, stage=ingestion_stage) 
            else:
                success = merge_result(scrapped_df)
                if success:
                    #wm.mark_success(amazon_org_id, today,0)
                    return None
                else:
                    wm.mark_failed(amazon_org_id, "MERGE into raw_openings failed")
        except Exception as e:
            print(f"Error scraping Amazon: {e}")
            wm.mark_failed(amazon_org_id, f"Error scraping Amazon: {e}")
            raise
    else:
        print(f"Amazon scrape skipped — {today} already loaded")
        
    
    return None


if __name__ == "__main__":
    main()
