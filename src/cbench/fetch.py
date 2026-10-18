"""Download ContraDoc (NAACL 2024, Apache-2.0) at a pinned commit and verify it."""

import hashlib
import urllib.request

from cbench.paths import CONTRADOC_JSON

COMMIT = "d1c42d1e433595dd6e98fab7b669722cdc050b1d"
URL = f"https://raw.githubusercontent.com/ddhruvkr/CONTRADOC/{COMMIT}/ContraDoc.json"
SHA256 = "def2213b8c6314353614ddadd0c8bbefeb883a2ccd4ebfcacacf10e8754e158c"


def main() -> None:
    if CONTRADOC_JSON.exists() and _sha256(CONTRADOC_JSON.read_bytes()) == SHA256:
        print(f"already present: {CONTRADOC_JSON}")
        return
    with urllib.request.urlopen(URL, timeout=60) as response:
        body = response.read()
    digest = _sha256(body)
    if digest != SHA256:
        raise SystemExit(f"sha256 mismatch: expected {SHA256}, got {digest}")
    CONTRADOC_JSON.parent.mkdir(parents=True, exist_ok=True)
    CONTRADOC_JSON.write_bytes(body)
    print(f"wrote {CONTRADOC_JSON} ({len(body):,} bytes)")


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


if __name__ == "__main__":
    main()
