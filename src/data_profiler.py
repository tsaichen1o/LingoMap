"""
LingoMap Data Profiler
A module to analyze a pandas Series and generate a structured profile.
"""

import pandas as pd
from typing import Dict, Any
import json
import re

def is_potential_excel_date(series: pd.Series) -> bool:
    """
    A simple heuristic rule to determine if a numeric column is likely to be an Excel date sequence.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False
    # Date in Excel for Mac starts from 1 (1904-01-01)
    # We use a reasonable range to guess, e.g., from 1970 to 2050
    min_date_serial = 25569 # 1970-01-01
    max_date_serial = 54786 # 2050-01-01
    # Check if most values fall within this range
    return series.dropna().between(min_date_serial, max_date_serial).mean() > 0.8

def convert_excel_date(serial_number):
    """Convert an Excel serial number to a YYYY-MM-DD string."""
    try:
        # '1899-12-30' is the standard base date for handling Excel 1900 leap year bug
        return (pd.to_datetime('1899-12-30') + pd.to_timedelta(int(serial_number), 'D')).strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return None

def profile_column(column_name: str, series: pd.Series) -> Dict[str, Any]:
    """
    Analyzes a pandas Series and returns a structured dictionary of its profile.

    Args:
        column_name: The name of the column.
        series: The pandas Series object representing the column data.

    Returns:
        A dictionary containing the profile of the column.
    """
    profile = {
        "column_name": column_name,
        "total_count": len(series),
    }

    # 1. Basic statistics
    missing_count = int(series.isna().sum())
    profile["missing_values"] = missing_count
    profile["missing_percentage"] = round((missing_count / profile["total_count"]) * 100, 2) if profile["total_count"] > 0 else 0
    profile["unique_values_count"] = int(series.nunique())
    
    # 2. Data type inference
    profile["inferred_pandas_dtype"] = str(series.dtype)

    # 3. Semantic type inference
    semantic_type = "Unknown"
    is_date_col = False
    
    # --- This is the main modification ---
    # First check if it's a potential date sequence
    if is_potential_excel_date(series):
        semantic_type = "Date"
        is_date_col = True
    elif pd.api.types.is_numeric_dtype(series):
        if profile["unique_values_count"] == profile["total_count"] and profile["unique_values_count"] > 1000:
             semantic_type = "Identifier"
        elif profile["unique_values_count"] < 25:
            semantic_type = "Categorical (Code)"
        else:
            semantic_type = "Numeric"
    elif pd.api.types.is_string_dtype(series):
        # Check if the string content looks like a date
        if series.dropna().str.match(r'\d{1,2}/\d{1,2}/\d{4}').mean() > 0.8:
            semantic_type = "Date"
            is_date_col = True
        elif profile["total_count"] > 0 and (profile["unique_values_count"] / profile["total_count"]) < 0.5:
             semantic_type = "Categorical"
        else:
            semantic_type = "Text"
            
    profile["inferred_semantic_type"] = semantic_type

    # 4. Numeric features
    if semantic_type == "Numeric":
        stats = series.describe()
        profile["numeric_stats"] = {
            "min": float(stats.get("min", 0)),
            "max": float(stats.get("max", 0)),
            "mean": float(stats.get("mean", 0)),
            "std": float(stats.get("std", 0)),
        }

    # 5. Categorical features
    if "Categorical" in semantic_type:
        top_values = series.value_counts().head(5).to_dict()
        profile["top_5_values"] = {str(k): int(v) for k, v in top_values.items()}

    # 6. Provide data examples (date conversion)
    sample_values = []
    for item in series.dropna().head(5).tolist():
        if is_date_col:
            # If it's a date column, try to convert it
            converted_date = convert_excel_date(item)
            if converted_date:
                sample_values.append(converted_date)
            else: # If conversion fails, show the original value
                sample_values.append(item)
        else:
            sample_values.append(item)
            
    profile["sample_values"] = sample_values

    return profile

def main():
    """Main function to demonstrate how to use the profiler"""
    
    try:
        csv_file = "FDIC_Insured_Banks.csv"
        df = pd.read_csv(csv_file, low_memory=False)
        print(f"✅ Successfully read data file: {csv_file}")
        print(f"📊 Total {len(df)} rows, {len(df.columns)} columns")
    except FileNotFoundError:
        print(f"❌ Error: Data file '{csv_file}' not found. Please ensure it's in the same folder as the script.")
        return

    # --- Analyze all columns ---
    all_columns = list(df.columns)

    print("\n" + "="*50)
    print("🚀 Start generating column profiling report...")
    print("="*50)
    
    all_profiles = {}
    for column in all_columns:
        print(f"\n--- Analyze column: {column} ---")
        series = df[column]
        report = profile_column(column, series)
        all_profiles[column] = report
        
        # Use json.dumps to pretty print the output
        print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # Save the complete report to a JSON file
    output_file = "data_profile_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_profiles, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Complete report saved to: {output_file}")
    
    # Show summary statistics
    print("\n" + "="*50)
    print("📋 Column analysis summary")
    print("="*50)
    
    missing_data_columns = []
    categorical_columns = []
    numeric_columns = []
    identifier_columns = []
    
    for column, profile in all_profiles.items():
        if profile['missing_percentage'] > 0:
            missing_data_columns.append((column, profile['missing_percentage']))
        
        semantic_type = profile['inferred_semantic_type']
        if 'Categorical' in semantic_type:
            categorical_columns.append(column)
        elif semantic_type == 'Numeric':
            numeric_columns.append(column)
        elif semantic_type == 'Identifier':
            identifier_columns.append(column)
    
    print(f"📊 Column type distribution:")
    print(f"   - Categorical columns: {len(categorical_columns)}")
    print(f"   - Numeric columns: {len(numeric_columns)}")
    print(f"   - Identifier columns: {len(identifier_columns)}")
    print(f"   - Other columns: {len(all_columns) - len(categorical_columns) - len(numeric_columns) - len(identifier_columns)}")
    
    if missing_data_columns:
        print(f"\n⚠️ Columns with missing data:")
        for column, percentage in sorted(missing_data_columns, key=lambda x: x[1], reverse=True):
            print(f"   - {column}: {percentage}% missing")
    
    print(f"\n✅ Analysis completed!")

if __name__ == "__main__":
    main()