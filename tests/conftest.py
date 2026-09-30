import sys
from pathlib import Path

DOMAIN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DOMAIN_DIR
for p in (DOMAIN_DIR / "ControlRoom", DOMAIN_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
