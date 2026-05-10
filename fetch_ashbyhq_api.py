import requests
import uuid
from datetime import datetime, date,timezone
from bs4 import BeautifulSoup
from properties import  RAW_TABLE,COMPANIES,ingestion_stage,QUARANTINE_SCRAPE_URLS
from pyspark.sql import Row, DataFrame, functions as F, Window

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pipeline_watermark import WatermarkManager


today_date=date.today()
SOURCE_TYPE="ashbyhq"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

def parse_description(html: str) -> str:
    """Strip HTML tags from descriptionHtml and remove boilerplate after 'About OpenAI'."""
    if not html:
        return ""
    text = BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)
    marker = "About OpenAI"
    idx = text.find(marker)
    if idx != -1:
        text = text[:idx].strip()
    return text


def parse_posted_date(published_at: str) -> date | None:
    """Convert ISO datetime string to date."""
    try:
        return datetime.fromisoformat(published_at).date()
    except (ValueError, TypeError):
        return None


def log_scrape_error(org, scrape_dt, ats, url, error):
    """Insert a row into the quarantine table for failed scrapes."""
    spark.sql(f"""
        INSERT INTO {QUARANTINE_SCRAPE_URLS} (
            scrape_org_key, scrape_dt, ats_name, scrape_url, error
        ) VALUES (
            '{org}', '{scrape_dt}', '{ats}', '{url}', '{error}'
        )
    """)


def fetch_ashbyhq_jobs(wm) -> list[dict]:
    results = []
    

    for ats, slugs in COMPANIES.items():
        for slug in slugs:
            url = f"https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true"
            wm.register_org(slug, stage=ingestion_stage, seed_dt=today_date)
            search_result = wm.check_if_rerun(slug,stage=ingestion_stage,scrape_tried_dt=today_date)
            if search_result:
                print(f"skipping {slug} as already ingested {today_date}")
                continue
            try:
                response = session.get(url, headers=HEADERS, timeout=(5, 30))
                response.raise_for_status()

                data = response.json()

                if not isinstance(data, dict):
                    print("No JSON data")
                    continue

                jobs = data.get("jobs")
                if not isinstance(jobs, list):
                    print("Not a valid response")
                    continue

                if len(jobs) == 0:
                    print(f"Zero rows scraped — nothing to merge for {today_date}")
                    wm.mark_no_data(slug, stage=ingestion_stage)
                    continue

                
                for job in jobs:
                    # skip unlisted jobs
                    if not job.get("isListed", True):
                        continue
                        
                    address = (job.get("address") or {}).get("postalAddress") or {}
                    location_parts = filter(None, [
                        address.get("addressLocality"),
                        address.get("addressRegion"),
                        address.get("addressCountry"),
                    ])
                    location = ", ".join(location_parts) or job.get("location", "")

                    # salary from compensation if available
                    compensation = job.get("compensation") or {}
                    salary_summary = compensation.get("scrapeableCompensationSalarySummary")

                    # build description, appending salary when available
                    description = parse_description(job.get("descriptionHtml", ""))
                    if salary_summary:
                        description = f"{description}\n\nCompensation: {salary_summary}"

                    results.append({
                        "doc_id":           str(uuid.uuid4()),
                        "source_type":      SOURCE_TYPE,
                        "organization":     slug,
                        "job_id":           job.get("id"),
                        "job_url":          job.get("jobUrl"),
                        "job_title":        job.get("title"),
                        "department":       job.get("department"),
                        "team":             job.get("team"),
                        "location":         location,
                        "workplace_type":   job.get("workplaceType"),
                        "employment_type":  job.get("employmentType"),
                        "job_description":  description,
                        "salary_summary":   salary_summary,
                        "posted_date":      parse_posted_date(job.get("publishedAt")),
                        "ingested_at":      datetime.now(timezone.utc),
                        "file_dt":          today_date,
                    })

            except requests.exceptions.Timeout:
                log_scrape_error(slug, today_date, ats, url, "Timeout fetching url")
                

            except requests.exceptions.ConnectionError:
                log_scrape_error(slug, today_date, ats, url, "Connection error fetching url")

            except requests.exceptions.HTTPError:
                log_scrape_error(slug, today_date, ats, url, "HTTP error fetching url")

            except ValueError:
                log_scrape_error(slug, today_date, ats, url, "Value error fetching url")

    return spark.createDataFrame(results)   


def _dedupe_for_merge(df: DataFrame) -> DataFrame:
    """
    Collapse source-side duplicates before MERGE.

    Ashby returns one job once per company, so within-company duplicates are
    rare — but a company that appears under multiple ATS slugs in COMPANIES,
    or future re-runs that re-emit the same job in the same batch, would
    duplicate. Dropping NULL job_ids is also defensive: Ashby's API can
    omit `id` for unlisted/draft postings.
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


def merge_result(rowsDf: DataFrame ,wm) -> bool:
    try:
        before = rowsDf.count()
        rowsDf = _dedupe_for_merge(rowsDf)
        after  = rowsDf.count()
        if before != after:
            print(f"[ashby] dedup collapsed {before} → {after} source rows")

        rowsDf.createOrReplaceTempView("ashbyhq_source")
        spark.sql(f"""
            MERGE INTO {RAW_TABLE} AS T
            USING ashbyhq_source AS S
            ON  T.job_id              = S.job_id
            AND lower(T.organization) = lower(S.organization)
            WHEN MATCHED THEN UPDATE SET
                ingested_at = S.ingested_at
            WHEN NOT MATCHED THEN INSERT (
                doc_id, source_type, source_url, organization, job_id,
                title, job_location, job_description,
                ingested_at, file_dt, notional_job_posted_dt, scrape_org_key,
                employment_type, workspace_type,team_name
            ) VALUES (
                S.doc_id, S.source_type, S.job_url,
                lower(S.organization), S.job_id,
                S.job_title, S.location,  S.job_description,
                S.ingested_at, S.file_dt, S.posted_date,
                lower(S.organization),
                S.employment_type, S.workplace_type, S.team
            )
        """)
        print(f"AshbyHq MERGE complete -> {RAW_TABLE}")
        return True
    except Exception as e:
        print(f"[ashby] MERGE failed: {type(e).__name__}: {e}")
        return False


def main()->None:
    try:
        wm    = WatermarkManager(spark)
        scrapped_df = fetch_ashbyhq_jobs(wm)
        if not scrapped_df.isEmpty():
            
            print(f"Opening returned from hashbyhq complete -> {scrapped_df.count()} rows returned")
            success = merge_result(scrapped_df,wm)
            if success:
                print(f"AshbyHq MERGE complete -> {RAW_TABLE}")
                # wm.mark_success(google_org_id, today,0)
            
                
    except Exception as e:
        print(f"Error fetching Ashbyhq api: {e}")
        return None

        
    return None


if __name__ == "__main__":
    main()
    
    
