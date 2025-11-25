from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..utils.logging import get_logger

try:
    import requests
    from tqdm.auto import tqdm
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def create_jwt(credentials_path: str, scope: str = "https://www.googleapis.com/auth/devstorage.read_only") -> str:
    """Create JWT for Google Cloud Storage authentication.

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


def download_file(session, file_url: str, local_filename: str, chunk_size_mb: int = 1, show_progress: bool = True) -> None:
    """Download a file from GCS with progress bar.

    Args:
        session: requests.Session with authentication
        file_url: Full URL to the file (mediaLink from GCS API)
        local_filename: Local path to save the file
        chunk_size_mb: Chunk size in MB for streaming download
        show_progress: Whether to show progress bar
    """
    response = session.get(file_url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        block_size = chunk_size_mb * 1024 * 1024  # Convert MB to bytes

        if show_progress and REQUESTS_AVAILABLE:
            with open(local_filename, 'wb') as f, tqdm(
                desc=os.path.basename(local_filename),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                start_time = time.time()
                downloaded = 0
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    downloaded += size
                    progress_bar.update(size)

                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        speed = downloaded / (1024 * 1024 * elapsed_time)  # MB/s
                        progress_bar.set_postfix(speed=f"{speed:.2f} MB/s", refresh=True)

            elapsed_time = time.time() - start_time
            logger = get_logger("gcs_download")
            logger.info("Downloaded %s", os.path.basename(local_filename))
            logger.info("Total size: %.2f MB", total_size / (1024 * 1024))
            if elapsed_time > 0:
                logger.info("Average speed: %.2f MB/s", downloaded / (1024 * 1024 * elapsed_time))
        else:
            # Simple download without progress bar
            with open(local_filename, 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
    else:
        raise RuntimeError(f"Failed to download {file_url}: HTTP {response.status_code}")


def list_gcs_files(bucket: str, prefix: str, access_token: str, proxy: Optional[str] = None) -> List[Dict[str, str]]:
    """List files in GCS bucket.

    Args:
        bucket: GCS bucket name
        prefix: Prefix to filter files (e.g., "exports/patents/")
        access_token: OAuth2 access token
        proxy: SOCKS5 proxy address

    Returns:
        List of file metadata dicts with 'name' and 'mediaLink' keys
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library is required for real GCS download. Install with: pip install requests")

    session = requests.Session()
    if proxy:
        session.proxies = {'http': f'socks5h://{proxy}', 'https': f'socks5h://{proxy}'}
    session.headers.update({"Authorization": f"Bearer {access_token}"})

    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    params = {'prefix': prefix}
    response = session.get(url, params=params)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to list GCS files: HTTP {response.status_code}\n{response.text}")

    files = response.json().get('items', [])
    return files


def run_download(
    exported_csv: str,
    raw_dir: str,
    *,
    dry_run: bool = True,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """Download CSV files from GCS.

    In dry-run mode: Simply verify exported CSV exists.
    In real mode: Download files from GCS bucket to local directory.

    Args:
        exported_csv: GCS URI (e.g., "gs://bucket/path/*.csv") or local path (dry-run)
        raw_dir: Local directory to save downloaded files
        dry_run: If True, skip actual download; if False, use real GCS API
        config: Configuration dict with gcs, auth, and processing keys

    Returns:
        Path to first downloaded CSV file or exported_csv (dry-run)
    """
    logger = get_logger("gcs_download")
    out_dir = Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        # Original dry-run behavior: just verify file exists
        p = Path(exported_csv)
        if not p.exists():
            raise FileNotFoundError(exported_csv)
        logger.info("Downloaded CSV available: %s", p)
        return str(p)

    # Real mode: Download from GCS
    if not config:
        raise ValueError("config parameter is required for real mode (dry_run=False)")

    if not REQUESTS_AVAILABLE:
        raise ImportError(
            "requests and tqdm libraries are required for real GCS download. "
            "Install with: pip install requests tqdm"
        )

    gcs_cfg = config.get("gcs", {})
    auth_cfg = config.get("auth", {})
    proc_cfg = config.get("processing", {})

    # Parse GCS URI
    if not exported_csv.startswith("gs://"):
        raise ValueError(f"Expected GCS URI (gs://...), got: {exported_csv}")

    # Extract bucket and prefix from GCS URI
    # Format: gs://bucket/path/file_*.csv
    gcs_path = exported_csv[5:]  # Remove "gs://"
    bucket_name = gcs_path.split('/')[0]
    prefix = '/'.join(gcs_path.split('/')[1:]).replace('*', '')  # Remove wildcard
    # Get directory path without filename pattern
    prefix = '/'.join(prefix.split('/')[:-1]) + '/' if '/' in prefix else ''

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
    logger.info("Creating JWT for GCS authentication...")
    jwt = create_jwt(credentials_path)

    logger.info("Getting access token...")
    access_token = get_access_token(jwt, proxy=proxy)
    if not access_token:
        raise RuntimeError("Failed to obtain access token")

    # List files in GCS bucket
    logger.info("Listing files in gs://%s/%s", bucket_name, prefix)
    files = list_gcs_files(bucket_name, prefix, access_token, proxy=proxy)
    logger.info("Found %d files in the bucket", len(files))

    # Download CSV files
    session = requests.Session()
    if proxy:
        session.proxies = {'http': f'socks5h://{proxy}', 'https': f'socks5h://{proxy}'}
    session.headers.update({"Authorization": f"Bearer {access_token}"})

    chunk_size_mb = proc_cfg.get("chunk_size_mb", 1)
    show_progress = proc_cfg.get("show_progress", True)

    downloaded_files = []
    for file in files:
        if file['name'].endswith('.csv'):
            file_url = file['mediaLink']
            local_filename = out_dir / os.path.basename(file['name'])

            logger.info("Starting download of %s...", file['name'])
            download_file(session, file_url, str(local_filename), chunk_size_mb, show_progress)
            downloaded_files.append(str(local_filename))

    if not downloaded_files:
        raise RuntimeError(f"No CSV files found in gs://{bucket_name}/{prefix}")

    logger.info("Downloaded %d CSV files to %s", len(downloaded_files), raw_dir)
    return downloaded_files[0]  # Return first file path