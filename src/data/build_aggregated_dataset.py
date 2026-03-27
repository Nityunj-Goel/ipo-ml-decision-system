from data_collection import *
from src.utils.utils import find_project_root

root = find_project_root()
cleaned_dir_base = root / "data" / "filtered"
aggregated_dir_base = root / "data" / "aggregated"

def build():
    output_csv_path = f'{aggregated_dir_base}'
    output_csv_file = 'dataset'

    print(f'Aggregating raw data from {cleaned_dir_base} into {output_csv_path}.csv and {output_csv_file}.json')

    filter_and_save_json_files()
    deduplicate_by_id(
        f'{cleaned_dir_base}/161ipo-key-financial-details-title-yyyy.json',
        f'{cleaned_dir_base}/161ipo-key-financial-details-title-yyyy_deduplicated.json'
    )
    parse_nse_data(
        f'{cleaned_dir_base}/id_to_nse_symbol_mapping.json',
        f'{cleaned_dir_base}/nseIpoData.json',
        f'{cleaned_dir_base}/nseAggregatedData.json',
    )
    merge_json_files_to_json_and_csv(
        [
            f'{cleaned_dir_base}/nseAggregatedData.json',
            f'{cleaned_dir_base}/gmp_data.json',
            f'{cleaned_dir_base}/162ipo-key-performance-indicator-kpi-title-yyyy.json',
            f'{cleaned_dir_base}/161ipo-key-financial-details-title-yyyy.json',
            f'{cleaned_dir_base}/98ipo_report_listing_day_gain.json'
        ],
        f'{aggregated_dir_base}/dataset.json',
        output_csv_path, output_csv_file
    )

    print('Aggregation complete!')

    return output_csv_path

if __name__ == '__main__':
    _ = build()