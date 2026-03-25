import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    df = df.copy()

    required_cols = [
        "CustomerType",
        "TrafficSource",
        "GeographicRegion",
        "PurchaseCompleted",
        "SpecialDayProximity",
        "ExitRate",
        "PageValue",
        "BounceRate",
        "ProductPageTime",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["CustomerType"] = (
        df["CustomerType"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "returning_visitor": "Returning_Visitor",
                "returning visitor": "Returning_Visitor",
                "new_visitor": "New_Visitor",
                "new visitor": "New_Visitor",
                "other": "Other",
                "": "Other",
                "nan": "Other",
                "none": "Other",
                "null": "Other",
                "unknown": "Other",
            }
        )
    )

    valid_customer_types = {"Returning_Visitor", "New_Visitor", "Other"}
    df["CustomerType"] = df["CustomerType"].apply(
        lambda x: x if x in valid_customer_types else "Other"
    )

    df.loc[df["BounceRate"] < 0, "BounceRate"] = pd.NA
    df.loc[df["ProductPageTime"] < 0, "ProductPageTime"] = pd.NA
    df.loc[df["ExitRate"] < 0, "ExitRate"] = pd.NA
    df.loc[df["PageValue"] < 0, "PageValue"] = pd.NA
    df.loc[df["SpecialDayProximity"] < 0, "SpecialDayProximity"] = pd.NA

    cat_cols = ["CustomerType", "TrafficSource", "GeographicRegion"]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(
            {
                "": "Unknown",
                "nan": "Unknown",
                "None": "Unknown",
                "none": "Unknown",
                "null": "Unknown",
            }
        )
        df[col] = df[col].astype("category")

    bool_map = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }

    if df["PurchaseCompleted"].dtype != bool:
        df["PurchaseCompleted"] = (
            df["PurchaseCompleted"].astype(str).str.strip().str.lower().map(bool_map)
        )

    if df["PurchaseCompleted"].isna().any():
        raise ValueError(
            "PurchaseCompleted contains values that could not be converted to boolean."
        )

    df["PurchaseCompleted"] = df["PurchaseCompleted"].astype(bool)

    float_cols = [
        "SpecialDayProximity",
        "ExitRate",
        "PageValue",
        "BounceRate",
        "ProductPageTime",
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
