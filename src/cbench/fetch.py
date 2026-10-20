"""Download ContraDoc (NAACL 2024, Apache-2.0) at a pinned commit and verify it.

`--cache` also restores the published run caches into results/cache, so the report
rebuilds without API keys or NLI compute.
"""

import argparse
import hashlib
import io
import tarfile
import urllib.request

from cbench.paths import CACHE, CONTRADOC_JSON

COMMIT = "d1c42d1e433595dd6e98fab7b669722cdc050b1d"
URL = f"https://raw.githubusercontent.com/ddhruvkr/CONTRADOC/{COMMIT}/ContraDoc.json"
SHA256 = "def2213b8c6314353614ddadd0c8bbefeb883a2ccd4ebfcacacf10e8754e158c"

CACHE_URL = "https://github.com/nstandif/contradiction-bench/releases/download/cache-2026-10-19/results-cache.tar.gz"
CACHE_SHA256 = "708ec3dcc22a1cca811a1438884ce282dbff3aa03bda919e1c1e166929a7bd67"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", action="store_true", help="also restore the published run caches")
    args = parser.parse_args()
    fetch_contradoc()
    if args.cache:
        fetch_cache()


def fetch_contradoc() -> None:
    if CONTRADOC_JSON.exists() and _sha256(CONTRADOC_JSON.read_bytes()) == SHA256:
        print(f"already present: {CONTRADOC_JSON}")
        return
    body = _download(URL, SHA256)
    CONTRADOC_JSON.parent.mkdir(parents=True, exist_ok=True)
    CONTRADOC_JSON.write_bytes(body)
    print(f"wrote {CONTRADOC_JSON} ({len(body):,} bytes)")


def fetch_cache() -> None:
    body = _download(CACHE_URL, CACHE_SHA256)
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as archive:
        archive.extractall(CACHE.parent, filter="data")
    print(f"restored run caches into {CACHE}")


def _download(url: str, sha256: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:
        body = response.read()
    if (digest := _sha256(body)) != sha256:
        raise SystemExit(f"sha256 mismatch for {url}: expected {sha256}, got {digest}")
    return body


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


if __name__ == "__main__":
    main()
