import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(
    df: pd.DataFrame,
    target_col: str = "PurchaseCompleted",
    test_size: float = 0.2,
    random_state: int = 42):
    """
    Split dataset into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    test_size : float
        Proportion of dataset to include in test split
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.isnull().any():
        raise ValueError("Target column contains missing values.")

    # Stratified split (important for imbalanced classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test