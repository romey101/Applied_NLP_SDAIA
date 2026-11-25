
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "frontend" / "public" / "demo"

OUTPUT.mkdir(parents=True, exist_ok=True)

API = "http://127.0.0.1:8000/api"

ENDPOINTS = [
    "overview",
    "universe",
    "metrics",
]

for endpoint in ENDPOINTS:
    with urlopen(f"{API}/{endpoint}", timeout=60) as response:
        data = json.load(response)

    if endpoint == "universe":
        for i, point in enumerate(data["points"]):
            point["review_id"] = i + 1

        for topic in data["topics"]:
            topic["keywords"] = []

    output_file = OUTPUT / f"{endpoint}.json"

    output_file.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        ),
        encoding="utf-8"
    )

    print(f"Exported: {output_file.name}")

print("Demo export completed.")
