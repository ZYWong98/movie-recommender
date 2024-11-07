from typing import Callable
import pandas as pd

def check_inputs(func: Callable) -> Callable:
    """
    Decorator function to validate inputs for precision_at_k & recall_at_k.
    Ensures k is positive and required columns are present in the DataFrame.
    """
    def checker(df: pd.DataFrame, k: int = 3, y_test: str = 'y_actual', y_pred: str = 'y_recommended') -> float:
        # Check that k is a positive integer
        if k <= 0:
            raise ValueError(f"Value of k should be greater than 0, read in as: {k}")
        
        # Check that DataFrame is not empty
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        
        # Check that y_test & y_pred columns are in df
        if y_test not in df.columns:
            raise ValueError(f"Input DataFrame does not have a column named: {y_test}")
        if y_pred not in df.columns:
            raise ValueError(f"Input DataFrame does not have a column named: {y_pred}")
        
        return func(df, k, y_test, y_pred)
    
    return checker

@check_inputs
def precision_at_k(df: pd.DataFrame, k: int, y_test: str = 'y_actual', y_pred: str = 'y_recommended') -> float:
    """
    Calculates Precision@K for a DataFrame with boolean columns indicating relevance.

    Parameters:
        df      : pd.DataFrame - DataFrame containing boolean columns for y_test & y_pred.
        k       : int          - Number of top items to consider.
        y_test  : str          - Column name for actual user-relevant items.
        y_pred  : str          - Column name for recommended items.
        
    Returns:
        float: Precision@K score for the top k items.
    """       
    # Extract top k rows
    df_k = df.head(k)
    # Calculate the number of recommended items in top K
    denominator = df_k[y_pred].sum()
    # Calculate the number of relevant recommended items in top K
    numerator = df_k[df_k[y_pred] & df_k[y_test]].shape[0]
    # Return Precision@K
    return numerator / denominator if denominator > 0 else 0.0

@check_inputs
def recall_at_k(df: pd.DataFrame, k: int, y_test: str = 'y_actual', y_pred: str = 'y_recommended') -> float:
    """
    Calculates Recall@K for a DataFrame with boolean columns indicating relevance.

    Parameters:
        df      : pd.DataFrame - DataFrame containing boolean columns for y_test & y_pred.
        k       : int          - Number of top items to consider.
        y_test  : str          - Column name for actual user-relevant items.
        y_pred  : str          - Column name for recommended items.
        
    Returns:
        float: Recall@K score for the top k items.
    """    
    # Extract top k rows
    df_k = df.head(k)
    # Calculate the total number of relevant items in the DataFrame
    denominator = df[y_test].sum()
    # Calculate the number of relevant recommended items in top K
    numerator = df_k[df_k[y_pred] & df_k[y_test]].shape[0]
    # Return Recall@K
    return numerator / denominator if denominator > 0 else 0.0
