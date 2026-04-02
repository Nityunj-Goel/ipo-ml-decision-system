from src.data import build_aggregated_dataset, preprocessor

csv_file = build_aggregated_dataset.build()
# verify if unpacking works as expected
preprocessor.preprocess(*csv_file, mode='strict')

# Everything needs to be in train
# Create a dict of model_type with pipeline method to generalize code
def train(model_type, X_train, y_train):
    if model_type == "logistic":
        pipeline = get_logistic_pipeline()
    elif model_type == "tree":
        pipeline = get_tree_pipeline()

    pipeline.fit(X_train, y_train)

    return pipeline

# rename final file