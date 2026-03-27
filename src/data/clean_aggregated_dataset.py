from os import PathLike
from pathlib import Path
import pandas as pd
from src.utils.utils import save_dataframe_to_csv

redundant_cols = [
    '~Issue_Open_Date',
    'closing_date'
]

def clean(path: PathLike, csv_file_name: str):
    csv_file = Path.joinpath(path, f'{csv_file_name}.csv')
    print(f"Cleaning {csv_file}")

    # Drop useless columns
    df = pd.read_csv(csv_file)
    df = df.drop(columns=redundant_cols)

    csv_file_name = f'{csv_file_name}_cleaned'
    csv_file = Path.joinpath(path, f'{csv_file_name}.csv')
    save_dataframe_to_csv(df, csv_file)

    print('Cleaning complete')
    return path, csv_file_name
