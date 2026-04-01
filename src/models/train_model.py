from src.data import build_aggregated_dataset, preprocessor

csv_file = build_aggregated_dataset.build()
# verify if unpacking works as expected
preprocessor.preprocess(*csv_file, mode='strict')

# rename final file