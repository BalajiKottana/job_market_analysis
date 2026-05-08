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
from properties import RAW_TABLE,  openai_org_id,ingestion_stage,web_browser_agent,openai_careers_url

today=date.today()

def start_scraping():

    resp = requests.get(openai_careers_url, headers={"User-Agent": web_browser_agent}, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    print(f"The value present in soup is {soup}")


    return None

    


def main()->None:
    wm    = WatermarkManager(spark)
    wm.register_org(openai_org_id, stage=ingestion_stage, seed_dt=today)
    if wm.check_if_rerun(openai_org_id,ingestion_stage,today):
        return None
    try:
        scrapped_df=start_scraping()

    except Exception as e:
            print(f"Error scraping Google: {e}")
            #wm.mark_failed(google_org_id, f"Error scraping Google: {e}")
            raise
        

    # wm.status()
    return None

if __name__ == "__main__":
    main()  
  