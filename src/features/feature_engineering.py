import pandas as pd
from configs.feature_config import DERIVED_FEATURES, RAW_FEATURES

def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    qib, retail, nii = RAW_FEATURES['qib'], RAW_FEATURES['retail'], RAW_FEATURES['nii']
    qib_ratio, retail_ratio, nii_ratio = (
        DERIVED_FEATURES['qib_ratio'], DERIVED_FEATURES['retail_ratio'], DERIVED_FEATURES['nii_ratio'])
    total = df[RAW_FEATURES['total']].replace(0, pd.NA)
    gmp, is_gmp_missing = RAW_FEATURES['gmp'], DERIVED_FEATURES['is_gmp_missing']

    df[qib_ratio] = df[qib] / total
    df[qib_ratio] = df[qib_ratio].fillna(0)

    df[retail_ratio] = df[retail] / total
    df[retail_ratio] = df[retail_ratio].fillna(0)

    df[nii_ratio] = df[nii] / total
    df[nii_ratio] = df[nii_ratio].fillna(0)

    df[is_gmp_missing] = df[gmp].isna().astype(int)
    df[gmp] = df[gmp].fillna(0)

    return df