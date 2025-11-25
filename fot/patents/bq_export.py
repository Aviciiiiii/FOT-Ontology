from __future__ import annotations

import base64
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from ..utils.logging import get_logger


def create_jwt(credentials_path: str, scope: str = "https://www.googleapis.com/auth/bigquery") -> str:
    """Create JWT for Google Cloud authentication.

    Args:
        credentials_path: Path to service account JSON file
        scope: OAuth2 scope for the token

    Returns:
        Signed JWT string
    """
    with open(credentials_path, 'r') as f:
        credentials = json.load(f)

    client_email = credentials['client_email']
    private_key = credentials['private_key']

    header = {"alg": "RS256", "typ": "JWT"}
    current_time = int(time.time())
    payload = {
        "iss": client_email,
        "scope": scope,
        "aud": "https://oauth2.googleapis.com/token",
        "exp": current_time + 3600,
        "iat": current_time
    }

    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip('=')
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')

    to_sign = f"{header_b64}.{payload_b64}"
    command = f"echo -n '{to_sign}' | openssl dgst -sha256 -sign <(echo '{private_key}') | openssl base64 -A | tr '+/' '-_' | tr -d '='"
    signature = subprocess.check_output(command, shell=True, executable='/bin/bash').decode().strip()

    return f"{header_b64}.{payload_b64}.{signature}"


def get_access_token(jwt: str, proxy: Optional[str] = None) -> Optional[str]:
    """Get OAuth2 access token using JWT.

    Args:
        jwt: Signed JWT token
        proxy: SOCKS5 proxy address (e.g., "127.0.0.1:7778")

    Returns:
        Access token string or None on failure
    """
    proxy_flag = f"--socks5-hostname {proxy}" if proxy else ""
    curl_command = f"""
    curl {proxy_flag} -L 'https://oauth2.googleapis.com/token' \
      -H 'Content-Type: application/x-www-form-urlencoded' \
      -d 'grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer&assertion={jwt}'
    """
    result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        try:
            token_response = json.loads(result.stdout)
            return token_response.get('access_token')
        except json.JSONDecodeError:
            return None
    return None


def start_export_job(
    access_token: str,
    project_id: str,
    dataset_id: str,
    table_id: str,
    gcs_bucket: str,
    gcs_filename: str,
    selected_fields: list,
    proxy: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Start BigQuery export job to GCS.

    Args:
        access_token: OAuth2 access token
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID (format: "project.dataset")
        table_id: BigQuery table ID
        gcs_bucket: GCS bucket name
        gcs_filename: Output filename pattern (e.g., "exports/data_*.csv")
        selected_fields: List of field names to export
        proxy: SOCKS5 proxy address

    Returns:
        Job response dict or None on failure
    """
    job_config = {
        "configuration": {
            "extract": {
                "sourceTable": {
                    "projectId": project_id,
                    "datasetId": dataset_id,
                    "tableId": table_id
                },
                "destinationUris": [f"gs://{gcs_bucket}/{gcs_filename}"],
                "destinationFormat": "CSV",
                "fieldDelimiter": ",",
            }
        }
    }

    if selected_fields:
        job_config["configuration"]["extract"]["sourceTable"]["selectedFields"] = selected_fields

    data = json.dumps(job_config)
    proxy_flag = f"--socks5-hostname {proxy}" if proxy else ""

    curl_command = f"""
    curl {proxy_flag} -X POST 'https://bigquery.googleapis.com/bigquery/v2/projects/{project_id}/jobs' \
      -H 'Authorization: Bearer {access_token}' \
      -H 'Content-Type: application/json' \
      -d '{data}'
    """

    result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        try:
            response = json.loads(result.stdout)
            return response
        except json.JSONDecodeError:
            return None
    return None


def check_job_status(
    access_token: str,
    project_id: str,
    job_id: str,
    proxy: Optional[str] = None
) -> Optional[str]:
    """Check BigQuery job status.

    Args:
        access_token: OAuth2 access token
        project_id: GCP project ID
        job_id: BigQuery job ID
        proxy: SOCKS5 proxy address

    Returns:
        Job state string ("PENDING", "RUNNING", "DONE") or None on failure
    """
    proxy_flag = f"--socks5-hostname {proxy}" if proxy else ""
    curl_command = f"""
    curl {proxy_flag} -X GET 'https://bigquery.googleapis.com/bigquery/v2/projects/{project_id}/jobs/{job_id}' \
      -H 'Authorization: Bearer {access_token}'
    """

    result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        try:
            response = json.loads(result.stdout)
            return response.get("status", {}).get("state")
        except json.JSONDecodeError:
            return None
    return None


def run_export(
    raw_dir: str,
    *,
    dry_run: bool = True,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """Export BigQuery table to GCS.

    In dry-run mode: Simulate export by writing a tiny CSV in raw_dir.
    In real mode: Execute actual BigQuery export to GCS.

    Args:
        raw_dir: Directory for output (dry-run) or tracking (real mode)
        dry_run: If True, generate synthetic CSV; if False, use real BigQuery API
        config: Configuration dict with bigquery, gcs, auth, and processing keys

    Returns:
        Path to exported CSV (dry-run) or GCS URI (real mode)
    """
    logger = get_logger("bq_export")
    out_dir = Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        # Original dry-run behavior: generate synthetic CSV
        csv_path = out_dir / "patents_sample.csv"
        rows = [
            ["US-123-A1", "Bread making device and method", "A21C"],
            ["US-456-A1", "Kneading apparatus improvements", "A21C1"],
            ["US-789-A1", "Meat mincing cutter design", "A22C"],
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["publication_number", "title", "ipc_code"])
            w.writerows(rows)
        logger.info("Exported CSV rows=%d -> %s", len(rows), csv_path)
        return str(csv_path)

    # Real mode: Execute BigQuery export
    if not config:
        raise ValueError("config parameter is required for real mode (dry_run=False)")

    bq_cfg = config.get("bigquery", {})
    gcs_cfg = config.get("gcs", {})
    auth_cfg = config.get("auth", {})
    proc_cfg = config.get("processing", {})

    # Get credentials path from config or environment
    credentials_path = auth_cfg.get("credentials_path") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path or not Path(credentials_path).exists():
        raise FileNotFoundError(
            f"Credentials file not found. Set auth.credentials_path in config or "
            f"GOOGLE_APPLICATION_CREDENTIALS environment variable."
        )

    # Prepare proxy setting
    proxy = None
    if auth_cfg.get("use_proxy", False):
        proxy = f"{auth_cfg.get('proxy_host', '127.0.0.1')}:{auth_cfg.get('proxy_port', 7778)}"

    # Create JWT and get access token
    logger.info("Creating JWT for authentication...")
    jwt = create_jwt(credentials_path)

    logger.info("Getting access token...")
    access_token = get_access_token(jwt, proxy=proxy)
    if not access_token:
        raise RuntimeError("Failed to obtain access token")

    # Start export job
    project_id = bq_cfg.get("project_id", "fot-search")
    dataset_id = bq_cfg.get("dataset_id", "patents-public-data.patents")
    table_id = bq_cfg.get("table_id", "publications_202310")
    gcs_bucket = gcs_cfg.get("bucket", "fot-bucket")
    gcs_path = gcs_cfg.get("export_path", "exports/patents/")
    gcs_pattern = gcs_cfg.get("filename_pattern", "data_export_*.csv")
    gcs_filename = f"{gcs_path}{gcs_pattern}"
    selected_fields = bq_cfg.get("selected_fields", [])

    logger.info("Starting BigQuery export job...")
    logger.info("  Project: %s", project_id)
    logger.info("  Dataset: %s", dataset_id)
    logger.info("  Table: %s", table_id)
    logger.info("  Destination: gs://%s/%s", gcs_bucket, gcs_filename)

    job = start_export_job(
        access_token, project_id, dataset_id, table_id,
        gcs_bucket, gcs_filename, selected_fields, proxy=proxy
    )

    if not job:
        raise RuntimeError("Failed to start export job")

    job_id = job.get("jobReference", {}).get("jobId")
    logger.info("Export job started successfully. Job ID: %s", job_id)

    # Poll job status
    poll_interval = proc_cfg.get("job_poll_interval", 10)
    timeout = proc_cfg.get("job_timeout", 3600)
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Job polling timeout after {timeout} seconds")

        status = check_job_status(access_token, project_id, job_id, proxy=proxy)
        logger.info("Job status: %s", status)

        if status == "DONE":
            logger.info("Export job completed successfully")
            break
        elif status in ["PENDING", "RUNNING"]:
            time.sleep(poll_interval)
        else:
            raise RuntimeError(f"Unexpected job status: {status}")

    gcs_uri = f"gs://{gcs_bucket}/{gcs_filename}"
    return gcs_uri