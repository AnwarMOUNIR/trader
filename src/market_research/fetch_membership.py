"""Fetch baseline S&P 500 membership datasets with complete cryptographic provenance."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import httpx
import pandas as pd

SOURCES = [
    {
        "id": "fja05680_components_updated",
        "url": "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv",
        "filename": "fja05680_sp500_historical_components_updated.csv",
        "description": "Historical S&P 500 constituents by date (1996 to present), derived from Andreas Clenow and Wikipedia updates.",
        "license": "MIT License (fja05680/sp500)",
    },
    {
        "id": "fja05680_ticker_start_end",
        "url": "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv",
        "filename": "fja05680_sp500_ticker_start_end.csv",
        "description": "Calculated first and last date of each ticker in S&P 500.",
        "license": "MIT License (fja05680/sp500)",
    },
    {
        "id": "fja05680_changes_since_2019",
        "url": "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_changes_since_2019.csv",
        "filename": "fja05680_changes_since_2019.csv",
        "description": "S&P 500 constituent additions and removals logged since 2019.",
        "license": "MIT License (fja05680/sp500)",
    },
    {
        "id": "fja05680_license",
        "url": "https://raw.githubusercontent.com/fja05680/sp500/master/LICENSE",
        "filename": "fja05680_LICENSE.txt",
        "description": "License file for fja05680 repository.",
        "license": "MIT License",
    },
    {
        "id": "datasets_constituents",
        "url": "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
        "filename": "datasets_sp500_constituents.csv",
        "description": "Curated S&P 500 list with CIK, GICS, and Date added (including 1957-03-04 for founding members).",
        "license": "Open Data Commons Public Domain Dedication and License (PDDL)",
    },
]

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def fetch_and_record():
    out_dir = Path("data/metadata/sources")
    prov_dir = Path("data/provenance")
    out_dir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Market-Research-SP500-Auditor/1.0 (Research)"}
    provenance_records = []

    with httpx.Client(headers=headers, timeout=30.0, follow_redirects=True) as client:
        for src in SOURCES:
            print(f"Downloading {src['id']} from {src['url']}...")
            try:
                resp = client.get(src['url'])
                resp.raise_for_status()
                content = resp.content
                file_hash = sha256_bytes(content)
                target_path = out_dir / src['filename']
                with open(target_path, "wb") as f:
                    f.write(content)
                
                record = {
                    "source_id": src["id"],
                    "source_url": src["url"],
                    "local_path": str(target_path.as_posix()),
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "sha256": file_hash,
                    "byte_size": len(content),
                    "license": src["license"],
                    "description": src["description"],
                    "status": "SUCCESS",
                }
                provenance_records.append(record)
                print(f"  Saved {src['filename']} ({len(content):,} bytes, SHA256: {file_hash[:12]}...)")
            except Exception as e:
                print(f"  Error fetching {src['id']}: {e}")
                provenance_records.append({
                    "source_id": src["id"],
                    "source_url": src["url"],
                    "error": str(e),
                    "status": "FAILED",
                })

        # Fetch Wikipedia directly
        print("Fetching Wikipedia List of S&P 500 companies...")
        wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            resp = client.get(wiki_url)
            resp.raise_for_status()
            wiki_html = resp.content
            wiki_hash = sha256_bytes(wiki_html)
            wiki_target = out_dir / "wikipedia_sp500.html"
            with open(wiki_target, "wb") as f:
                f.write(wiki_html)

            # Parse tables with pandas using StringIO
            import io
            tables = pd.read_html(io.StringIO(wiki_html.decode('utf-8')))
            curr_df = tables[0]
            changes_df = tables[1] if len(tables) > 1 else pd.DataFrame()

            curr_csv_path = out_dir / "wikipedia_current_constituents.csv"
            changes_csv_path = out_dir / "wikipedia_changes.csv"
            curr_df.to_csv(curr_csv_path, index=False)
            changes_df.to_csv(changes_csv_path, index=False)

            provenance_records.append({
                "source_id": "wikipedia_sp500_page",
                "source_url": wiki_url,
                "local_path": str(wiki_target.as_posix()),
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": wiki_hash,
                "byte_size": len(wiki_html),
                "license": "Creative Commons Attribution-ShareAlike License 4.0",
                "description": "Direct Wikipedia page capture with current members and selected changes tables.",
                "parsed_files": [
                    str(curr_csv_path.as_posix()),
                    str(changes_csv_path.as_posix()),
                ],
                "status": "SUCCESS",
            })
            print(f"  Parsed Wikipedia: {len(curr_df)} current members, {len(changes_df)} historical changes logged.")
        except Exception as e:
            print(f"  Error fetching Wikipedia: {e}")
            provenance_records.append({
                "source_id": "wikipedia_sp500_page",
                "source_url": wiki_url,
                "error": str(e),
                "status": "FAILED",
            })

    # Save master provenance register
    prov_file = prov_dir / "sources_provenance.json"
    with open(prov_file, "w") as f:
        json.dump(provenance_records, f, indent=2)

    print(f"\nSaved provenance metadata to {prov_file}")

if __name__ == "__main__":
    fetch_and_record()
