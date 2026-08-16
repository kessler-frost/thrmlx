"""Print a machine-readable summary of the pinned upstream parity ledger."""

import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "tests" / "upstream_parity" / "manifest.json"


def main() -> None:
    """Print the current count of planned and verified translated objectives."""

    manifest = json.loads(MANIFEST_PATH.read_text())
    statuses = [entry["status"] for entry in manifest["tests"]]
    green = statuses.count("green")
    planned = statuses.count("planned")
    report = {
        "complete": green == len(statuses),
        "green": green,
        "planned": planned,
        "total": len(statuses),
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
