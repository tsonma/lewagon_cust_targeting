import pandas as pd

def load_data(filepath="../data/bank-full.csv"):
    """
    Loads the bank dataset and processes the target variable.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        X (pd.DataFrame): Feature set with 'y' and 'duration' dropped.
        y (pd.Series): Binary target variable.
    """
    df_bank = pd.read_csv(filepath, sep=None, engine="python", header=0)
    X = df_bank.drop(['y', 'duration'], axis=1)
    y = df_bank['y'].map({'no': 0, 'yes': 1})
    return X, y

from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.20, val_size=0.25, random_state=42):
    """
    Splits the data into train, validation, and test sets.

    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        test_size (float): Proportion for the test set (default 0.20).
        val_size (float): Proportion of the temp set to use as validation (default 0.25).
        random_state (int): Random seed.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: Split datasets.
    """
    # First split: train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
