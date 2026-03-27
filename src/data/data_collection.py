import json
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime
import csv

def filter_and_save_json_files():
    # Paths relative to this file
    base_dir = Path(__file__).resolve().parents[2]

    raw_dir = base_dir / "data" / "raw"
    filtered_dir = base_dir / "data" / "filtered"

    filtered_dir.mkdir(parents=True, exist_ok=True)

    # filename -> properties to keep
    files_to_clean = {
        "25ipo-listing-date-check-status-price-bse-nse.json":
            [
                "~id",
                "Company",
                "NSE Symbol",
            ],
        "98ipo_report_listing_day_gain.json":
            [
                "~id",
                "Company",
                "Issue Price (Rs.)",
                "QIB (x)",
                "NII (x)",
                "Retail (x)",
                "Total (x)",
                "~IPO_Listing_Date",
                "Open Price on Listing (Rs.)",
                "year"
            ],
        "161ipo-key-financial-details-title-yyyy.json":
            [
                "~id",
                "Company",
                "~Issue_Open_Date",
                "Issue Amount (Rs.cr.)",
                "year",
                "Assets (Rs.cr.)",
                "Revenue (Rs.cr.)",
                "Profit After Tax (Rs.cr.)",
                "Ebitda (Rs.cr.)",
                "Net Worth (Rs.cr.)",
                "Reserves and Surplus (Rs.cr.)",
                "Total Borrowing (Rs.cr.)"
            ],
        "162ipo-key-performance-indicator-kpi-title-yyyy.json":
            [
                "~id",
                "Company",
                "year",
                "ROE",
                "ROCE",
                "RoNW",
                "PAT Margin %",
                "Price to Book Value",
                "EPS (Rs.) Pre-IPO",
                "P/E (x) Pre-IPO"
            ],
        "gmp_data.json":
            [
                "ipo_id",
                "company",
                "closing_date",
                "gmp_on_close"
            ],
        "nseIpoData.json":
            [
                "company",
                "symbol",
                "ipoStartDate",
                "ipoEndDate",
                "priceRange",
            ]
    }

    for filename, props in files_to_clean.items():

        input_path = raw_dir / filename

        if not input_path.exists():
            print(f"Warning: {filename} not found in raw folder")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cleaned_data = []

        for obj in data:
            cleaned_obj = {k: obj.get(k) for k in props}
            cleaned_data.append(cleaned_obj)

        output_path = filtered_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=2)

        print(f"Filtered {filename} -> {output_path}")

def deduplicate_by_id(input_file_path, output_file_path):
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    groups = defaultdict(list)

    for obj in data:
        groups[obj["~id"]].append(obj)

    unique_ids_initial = len(groups)
    duplicated_ids = 0
    retained_from_duplicates = 0
    discarded_ids = 0

    result = []

    for _id, objs in groups.items():

        if len(objs) == 1:
            result.append(objs[0])
            continue

        duplicated_ids += 1

        first = objs[0]
        all_identical = all(o == first for o in objs)

        if all_identical:
            result.append(first)
            retained_from_duplicates += 1
        else:
            discarded_ids += 1

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Initial unique ids:", unique_ids_initial)
    print("Duplicated ids:", duplicated_ids)
    print("Retained duplicate ids:", retained_from_duplicates)
    print("Discarded duplicate ids:", discarded_ids)

def parse_nse_data(mapping_file, nse_file, output_file):

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    with open(nse_file, "r", encoding="utf-8") as f:
        nse_data = json.load(f)

    nse_lookup = {obj["symbol"]: obj for obj in nse_data}

    result = []

    date_format = "%d-%b-%Y"

    pattern_to = re.compile(
        r"Rs\.?\s*(\d+)\s*to\s*Rs\.?\s*(\d+)",
        re.IGNORECASE
    )

    pattern_to_single_rs = re.compile(
        r"Rs\.?\s*(\d+)\s*to\s*(\d+)",
        re.IGNORECASE
    )

    pattern_space = re.compile(
        r"Rs\.?\s*(\d+)\s+Rs\.?\s*(\d+)",
        re.IGNORECASE
    )

    for m in mapping_data:

        symbol = m.get("NSE Symbol")
        nse_obj = nse_lookup.get(symbol)

        start_date = None
        end_date = None
        price_low = None
        price_high = None

        if nse_obj:

            start_raw = (nse_obj.get("ipoStartDate") or "").strip()
            end_raw = (nse_obj.get("ipoEndDate") or "").strip()

            try:
                start_date = datetime.strptime(start_raw, date_format).date()
                end_date = datetime.strptime(end_raw, date_format).date()
            except Exception:
                print("Invalid date format object:")
                print(nse_obj)
                raise ValueError("Date format must be like 15-JUL-2020")

            if start_date > end_date:
                print("Start date after end date:")
                print(nse_obj)
                raise ValueError("ipoStartDate cannot be after ipoEndDate")

            price_range = (nse_obj.get("priceRange") or "").strip()

            match = pattern_to.fullmatch(price_range)

            if not match:
                match = pattern_to_single_rs.fullmatch(price_range)

            if not match:
                match = pattern_space.fullmatch(price_range)

            if not match:
                print("Invalid priceRange object:")
                print(nse_obj)
                raise ValueError(
                    "priceRange must be like 'Rs 180 to Rs 186' or 'Rs.130 Rs.140'"
                )

            price_low = int(match.group(1))
            price_high = int(match.group(2))

            if price_low > price_high:
                print("Invalid price band:")
                print(nse_obj)
                raise ValueError(
                    "price_band_low cannot be greater than price_band_high"
                )

            start_date = start_date.isoformat()
            end_date = end_date.isoformat()

        result.append({
            "~id": m["~id"],
            "Company": m["Company"],
            "ipoStartDate": start_date,
            "ipoEndDate": end_date,
            "price_band_low": price_low,
            "price_band_high": price_high
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    assert len(result) == len(mapping_data)

def merge_json_files_to_json_and_csv(input_files, output_json, output_csv):

    records = {}
    all_fields = set()

    for file in input_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for obj in data:

            # normalize id field
            _id = obj.get("~id") or obj.get("ipo_id")
            company = obj.get("Company")

            if _id is None:
                continue

            key = _id

            if key not in records:
                records[key] = {"~id": _id, "Company": company}

            for k, v in obj.items():

                if k == "ipo_id":
                    k = "~id"

                if k.lower() == "company":
                    k = "Company"

                records[key][k] = v
                all_fields.add(k)

            all_fields.add("~id")
            all_fields.add("Company")

    all_fields = sorted(all_fields)

    final = []

    for obj in records.values():
        row = {field: obj.get(field, None) for field in all_fields}

        issue_price = row.get("Issue Price (Rs.)")
        listing_price = row.get("Open Price on Listing (Rs.)")

        def is_numeric(v):
            try:
                float(v)
                return True
            except:
                return False

        if not is_numeric(issue_price) or not is_numeric(listing_price):
            continue

        final.append(row)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(final)
