"""Data preparation utilities for TL-SQL pipelines """

from typing import Dict

import pandas as pd

from rllm.data import TableData
from rllm.types import ColType


def convert_dataframe_to_tabledata(
    df: pd.DataFrame,
    target_col: str,
    col_types: Dict[str, ColType]
) -> TableData:
    """Convert DataFrame to TableData object"""
    return TableData(
        df=df,
        col_types=col_types,
        target_col=target_col
    )


def remove_overlap_rows(
    df_to_remove_from: pd.DataFrame,
    df_reference: pd.DataFrame,
    dataset_name: str,
    reference_name: str
) -> pd.DataFrame:
    """Remove overlapping rows between two DataFrames"""
    common_cols = set(df_to_remove_from.columns) & set(df_reference.columns)
    reference_rows = set(
        tuple(row) for row in df_reference[list(common_cols)].values
    )

    mask = [
        tuple(row[col] for col in common_cols) not in reference_rows
        for _, row in df_to_remove_from.iterrows()
    ]

    return df_to_remove_from[mask].reset_index(drop=True)






