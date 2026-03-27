from src.data import build_aggregated_dataset, clean_aggregated_dataset

csv_file = build_aggregated_dataset.build()
# verify if unpacking works as expected
clean_aggregated_dataset.clean(*csv_file)

# rename final file