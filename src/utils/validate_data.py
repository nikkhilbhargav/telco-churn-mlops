import great_expectations as ge
import pandas as pd
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Validate Telco Customer Churn dataset using Great Expectations.
    """

    print("🔍 Starting data validation with Great Expectations...")

    # ===============================
    # FIX: Convert TotalCharges to numeric
    # ===============================
    df = df.copy()

    df["TotalCharges"] = (
        df["TotalCharges"]
        .replace(" ", pd.NA)
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing values with 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Create GE Dataset AFTER conversion
    ge_df = ge.dataset.PandasDataset(df)

    # ===============================
    # Schema Validation
    # ===============================
    print("   📋 Validating schema and required columns...")

    required_columns = [
        "customerID",
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    for col in required_columns:
        ge_df.expect_column_to_exist(col)

    ge_df.expect_column_values_to_not_be_null("customerID")

    # ===============================
    # Business Validation
    # ===============================
    print("   💼 Validating business logic constraints...")

    ge_df.expect_column_values_to_be_in_set(
        "gender",
        ["Male", "Female"]
    )

    ge_df.expect_column_values_to_be_in_set(
        "Partner",
        ["Yes", "No"]
    )

    ge_df.expect_column_values_to_be_in_set(
        "Dependents",
        ["Yes", "No"]
    )

    ge_df.expect_column_values_to_be_in_set(
        "PhoneService",
        ["Yes", "No"]
    )

    ge_df.expect_column_values_to_be_in_set(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year",
        ],
    )

    ge_df.expect_column_values_to_be_in_set(
        "InternetService",
        [
            "DSL",
            "Fiber optic",
            "No",
        ],
    )

    # ===============================
    # Numeric Validation
    # ===============================
    print("   📊 Validating numeric ranges...")

    ge_df.expect_column_values_to_be_between(
        "tenure",
        min_value=0,
        max_value=120,
    )

    ge_df.expect_column_values_to_be_between(
        "MonthlyCharges",
        min_value=0,
        max_value=200,
    )

    ge_df.expect_column_values_to_be_between(
        "TotalCharges",
        min_value=0,
    )

    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")
    ge_df.expect_column_values_to_not_be_null("TotalCharges")

    # ===============================
    # Consistency Check
    # ===============================
    print("   🔗 Validating data consistency...")

    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95,
    )

    # ===============================
    # Run Validation
    # ===============================
    print("   ⚙️ Running validation...")

    results = ge_df.validate()

    failed_expectations = []

    for r in results["results"]:
        if not r["success"]:
            failed_expectations.append(
                r["expectation_config"]["expectation_type"]
            )

    total = len(results["results"])
    passed = sum(r["success"] for r in results["results"])

    if results["success"]:
        print(f"✅ Validation Passed ({passed}/{total})")
    else:
        print(f"❌ Validation Failed ({passed}/{total})")
        print(failed_expectations)

    return results["success"], failed_expectations