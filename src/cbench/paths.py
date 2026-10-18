from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
WORK = ROOT / "data" / "work"
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"

CONTRADOC_JSON = RAW / "ContraDoc.json"
