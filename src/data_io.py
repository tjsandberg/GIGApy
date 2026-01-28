"""
Data I/O utilities for loading databases and saving results.
"""
import numpy as np
import pandas as pd
from datetime import datetime

def load_database_with_dtypes(dbase_file, dbusage_file):
    """
    Load database with proper dtypes from usage file.
    
    Parameters:
    -----------
    dbase_file : str
        Path to the database CSV file
    dbusage_file : str
        Path to the database usage Excel file
        
    Returns:
    --------
    df : DataFrame
        Full database with proper dtypes
    dfUsage : DataFrame
        Usage specification dataframe
    """
    # Load Feature Usage
    print(f"\nLOADING Feature Usage FROM {dbusage_file}")
    dfUsage = pd.read_excel(dbusage_file)
    feature_dtypes = dict(zip(dfUsage["Feature"], dfUsage["dtype"]))

    # Replace Int64/Int32/etc with float64/float32 for reading
    dtype_for_reading = {}
    int_to_float_map = {
        'Int64': 'float64', 
        'Int32': 'float32', 
        'Int16': 'float32', 
        'Int8': 'float32'
    }

    for col, dtype in feature_dtypes.items():
        if dtype in int_to_float_map:
            dtype_for_reading[col] = int_to_float_map[dtype]
        else:
            dtype_for_reading[col] = dtype

    # Load database with compatible dtypes
    print(f"\nLOADING DATA FROM {dbase_file}")
    df = pd.read_csv(dbase_file, sep=',', dtype=dtype_for_reading)
    print(f"Dataset shape: {df.shape}")

    # Convert float back to nullable Int after reading
    for col, dtype in feature_dtypes.items():
        if col in df.columns and dtype.startswith('Int'):
            df[col] = df[col].astype(dtype)

    return df, dfUsage

def save_results_to_excel(filename, sheets_dict):
    """
    Save multiple dataframes to an Excel file with multiple sheets.
    
    Parameters:
    -----------
    filename : str
        Output filename (should end in .ods or .xlsx)
    sheets_dict : dict
        Dictionary mapping sheet names to DataFrames
        Example: {'Sheet1': df1, 'Sheet2': df2}
    """
    with pd.ExcelWriter(filename) as writer:
        for sheet_name, df in sheets_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Results saved to '{filename}'")

def create_notes_dataframe(notes_dict):
    """
    Create a standardized notes DataFrame from a dictionary.
    
    Parameters:
    -----------
    notes_dict : dict
        Dictionary of field names to values
        Example: {'Target': 'temp', 'Input File': 'data.csv'}
        
    Returns:
    --------
    notes : DataFrame
        Two-column DataFrame with Field and Value columns
    """
    notes = pd.DataFrame({
        "Field": list(notes_dict.keys()),
        "Value": [str(v) for v in notes_dict.values()]
    })
    return notes

def generate_output_filename(scratch_dir, prefix, timestamp=True, extension='ods'):
    """
    Generate a unique output filename with optional timestamp.
    
    Parameters:
    -----------
    scratch_dir : str
        Directory for output files
    prefix : str
        Prefix for filename (e.g., target column name)
    timestamp : bool
        If True, include timestamp in filename
    extension : str
        File extension (default: 'ods')
        
    Returns:
    --------
    filename : str
        Full path to output file
    """
    if timestamp:
        now = datetime.now()
        filename = f"{scratch_dir}{prefix}_{now:%Y%m%d%H%M%S}.{extension}"
    else:
        filename = f"{scratch_dir}{prefix}.{extension}"
    
    return filename

def add_storm_maturity_features(df):
    """
    Add features indicating storm age/maturity using existing SampleNum.
    SampleNum is 0-indexed and handles gaps in observations correctly.
    """
    # Storm age in hours (SampleNum * 6 since samples are 6hr apart)
    df['storm_age_hours'] = df['SampleNum'] * 6
    df['storm_age_days'] = df['storm_age_hours'] / 24
    
    # Is this an early observation? (first 24 hours of tracking)
    df['is_early_storm'] = (df['storm_age_hours'] < 24).astype(int)
    
    print(f"\nStorm maturity features added:")
    print(f"  Age range: {df['storm_age_hours'].min()}-{df['storm_age_hours'].max()} hours")
    print(f"  Early storm samples: {df['is_early_storm'].sum()} / {len(df)}")
    
    return df

def prepare_hurricane_features_simplified(df, dfUsage, include_targets=False):
    """
    Simplified - just fill nulls, no derived features needed.
    Assumes SampleNum is marked as "input" in dbUsage.
    """
    print("\n" + "="*70)
    print("PREPARING HURRICANE FEATURES")
    print("="*70)
    
    # Fill historical lag nulls
    hist_cols = [col for col in df.columns if col.startswith('Hist')]
    null_count = df[hist_cols].isnull().sum().sum()
    for col in hist_cols:
        df[col] = df[col].fillna(0)
    print(f"Filled {null_count} historical nulls with 0")
    
    # Standard feature prep
    targetColumns = dfUsage["Feature"][(dfUsage["Usage"] == "target")]
    
    if include_targets:
        dropFeatures = dfUsage["Feature"][(dfUsage["Usage"] == "ignore")]
    else:
        dropFeatures = dfUsage["Feature"][(dfUsage["Usage"] == "target") | 
                                          (dfUsage["Usage"] == "ignore")]
    
    X = df.drop(columns=dropFeatures)
    X = X.select_dtypes(include=[np.number])
    
    # Handle other nulls
    remaining_nulls = X.isnull().sum()
    if remaining_nulls.sum() > 0:
        X = X.fillna(X.median())
        print(f"Filled {remaining_nulls.sum()} non-historical nulls with medians")
    
    # Convert Series to DataFrame for better Excel output
    remaining_nulls_df = pd.DataFrame({
        'Feature': remaining_nulls.index,
        'Null_Count': remaining_nulls.values
    })
    
    # Sort by null count (descending) and filter to only features with nulls
    remaining_nulls_df = remaining_nulls_df[remaining_nulls_df['Null_Count'] > 0]
    remaining_nulls_df = remaining_nulls_df.sort_values('Null_Count', ascending=False).reset_index(drop=True)
    
    print(f"\nFinal: {X.shape[1]} features, {X.shape[0]} samples")

    # Verify SampleNum is included
    if 'SampleNum' in X.columns:
        print(f"✓ SampleNum included (range: {X['SampleNum'].min()}-{X['SampleNum'].max()})")
    else:
        print("⚠ Warning: SampleNum not found - mark as 'input' in dbUsage")
    
    return X, targetColumns, null_count, remaining_nulls_df  # Return DataFrame instead of Series
