from typing import Literal

CleaningMode = Literal['strict', 'unstrict']

RAW_FEATURES = {
    "assets": "Assets (Rs.cr.)",
    "company": "Company",
    "eps": "EPS (Rs.) Pre-IPO",
    "ebitda": "Ebitda (Rs.cr.)",
    "issue_amount": "Issue Amount (Rs.cr.)",
    "issue_price": "Issue Price (Rs.)",
    "nii": "NII (x)",
    "net_worth": "Net Worth (Rs.cr.)",
    "listing_price": "Open Price on Listing (Rs.)",
    "pe": "P/E (x) Pre-IPO",
    "pat_margin": "PAT Margin %",
    "price_to_book_value": "Price to Book Value",
    "pat": "Profit After Tax (Rs.cr.)",
    "qib": "QIB (x)",
    "roce": "ROCE",
    "roe": "ROE",
    "reserves": "Reserves and Surplus (Rs.cr.)",
    "retail": "Retail (x)",
    "revenue": "Revenue (Rs.cr.)",
    "ronw": "RoNW",
    "total": "Total (x)",
    "total_borrowing": "Total Borrowing (Rs.cr.)",
    "closing_date": "closing_date",
    "gmp": "gmp_on_close",
    "ipo_end_date": "ipoEndDate",
    "ipo_start_date": "ipoStartDate",
    "price_band_high": "price_band_high",
    "price_band_low": "price_band_low",
    "year": "year",
    "listing_date": "~IPO_Listing_Date",
    "issue_open_date": "~Issue_Open_Date",
    "id": "~id",
}

DERIVED_FEATURES = {
    "qib_ratio": "qib_ratio",
    "retail_ratio": "retail_ratio",
    "nii_ratio": "nii_ratio",
    "log_issue_amount": "log_issue_amount",
    "log_total": "log_total",
    "is_gmp_missing": "is_gmp_missing"
}

# Model input
FINAL_FEATURES = [
    RAW_FEATURES["nii"],
    RAW_FEATURES["qib"],
    RAW_FEATURES["retail"],
    RAW_FEATURES["total"],
    DERIVED_FEATURES["qib_ratio"],
    DERIVED_FEATURES["retail_ratio"],
    DERIVED_FEATURES["nii_ratio"],
    RAW_FEATURES["year"],
    RAW_FEATURES["issue_amount"],
    RAW_FEATURES["price_band_high"],
    RAW_FEATURES["price_band_low"],
    RAW_FEATURES["gmp"],
    DERIVED_FEATURES['is_gmp_missing'],
]

TARGET = {
    "listing_price": RAW_FEATURES["listing_price"],
    "issue_price": RAW_FEATURES["issue_price"],
    "listing_gain_perc": "listing_gain_perc"
}