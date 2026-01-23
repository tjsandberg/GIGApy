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


def prepare_features(df, dfUsage, include_targets=False):
    """
    Prepare feature matrix X by dropping ignored/target columns.
    
    Parameters:
    -----------
    df : DataFrame
        Full database
    dfUsage : DataFrame
        Usage specification
    include_targets : bool
        If False, drop target columns. If True, keep them.
        
    Returns:
    --------
    X : DataFrame
        Feature matrix (numeric only)
    targetColumns : Index
        List of target column names
    """
    targetColumns = dfUsage["Feature"][(dfUsage["Usage"] == "target")]
    
    if include_targets:
        dropFeatures = dfUsage["Feature"][(dfUsage["Usage"] == "ignore")]
    else:
        dropFeatures = dfUsage["Feature"][(dfUsage["Usage"] == "target") | 
                                          (dfUsage["Usage"] == "ignore")]
    
    # Drop specified features
    X = df.drop(columns=dropFeatures)
    
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
        
    print(f"Final number of features: {X.shape[1]} after dropping non-numeric columns")
    print(f"Number of samples: {X.shape[0]}")
    
    return X, targetColumns


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
            df.to_excel(writer, sheet_name=sheet_name, index=True)
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
    
def calculate_data_quality_stats(df, X):
    """Calculate and report data quality statistics."""
    stats = {}
    
    # Null percentages
    null_pct = (X.isnull().sum() / len(X)) * 100
    stats['Features_>50pct_null'] = (null_pct > 50).sum()
    stats['Features_>20pct_null'] = (null_pct > 20).sum()
    stats['Avg_null_pct'] = null_pct.mean()
    
    # Temporal coverage if available
    if 'Year' in df.columns:
        stats['Year_range'] = f"{df['Year'].min()}-{df['Year'].max()}"
        stats['Years_covered'] = df['Year'].nunique()
    
    # Storm coverage
    if 'Desig' in df.columns:
        stats['Unique_storms'] = df['Desig'].nunique()
        stats['Avg_obs_per_storm'] = len(df) / df['Desig'].nunique()
    
    return stats

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


def add_lag_availability_indicators(df):
    """
    Add binary indicators for whether historical data is available.
    Uses 0-indexed SampleNum to determine which lags should exist.
    """
    lag_windows = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
    
    for lag in lag_windows:
        # Available if SampleNum >= required_samples
        # e.g., 24hr lag needs SampleNum >= 4 (samples 0,1,2,3 = 4 prior samples)
        required_samples = lag // 6
        df[f'has_hist_{lag}hr'] = (df['SampleNum'] >= required_samples).astype(int)
    
    # Summary
    print(f"\nLag availability indicators added for: {lag_windows}")
    for lag in lag_windows:
        available = df[f'has_hist_{lag}hr'].sum()
        print(f"  {lag}hr lag available: {available} / {len(df)} samples ({100*available/len(df):.1f}%)")
    
    return df


def prepare_hurricane_features_with_lags(df, dfUsage, include_targets=False, saveTLMdb=False):
    """
    Prepare features with intelligent handling of temporal lags.
    Uses existing SampleNum which correctly handles gaps in observations.
    """
    print("\n" + "="*70)
    print("PREPARING HURRICANE FEATURES WITH LAG HANDLING")
    print("="*70)
    
    # Add storm maturity features using SampleNum
    df = add_storm_maturity_features(df)
    
    # Add indicators for lag availability
    df = add_lag_availability_indicators(df)
    
    # Fill historical lag nulls
    hist_cols = [col for col in df.columns if col.startswith('Hist')]
    print(f"\nProcessing {len(hist_cols)} historical lag features...")
    
    null_counts_before = df[hist_cols].isnull().sum().sum()
    
    for col in hist_cols:
        df[col] = df[col].fillna(0)
    
    print(f"Filled {null_counts_before} historical lag nulls with 0")

    if saveTLMdb:
        outFileName = generate_output_filename("./tmp/", f"timeLagNulls")
        print(f"outFileName before replace: {outFileName}")
        outFileName = outFileName.replace("ods","csv")
        print(f"outFileName after replace: {outFileName}")
        df.to_csv(outFileName, index=False) 
        print(f"DB with time lag modifications saved to '{outFileName}'")
    
    # Standard feature preparation
    targetColumns = dfUsage["Feature"][(dfUsage["Usage"] == "target")]
    
    if include_targets:
        dropFeatures = dfUsage["Feature"][(dfUsage["Usage"] == "ignore")]
    else:
        dropFeatures = dfUsage["Feature"][(dfUsage["Usage"] == "target") | 
                                          (dfUsage["Usage"] == "ignore")]
    
    X = df.drop(columns=dropFeatures)
    X = X.select_dtypes(include=[np.number])
    
    # Handle remaining nulls (non-historical features)
    remaining_null_cols = X.columns[X.isnull().any()].tolist()
    
    if len(remaining_null_cols) > 0:
        print(f"\nHandling nulls in {len(remaining_null_cols)} non-historical features:")
        remaining_nulls = X[remaining_null_cols].isnull().sum()
        print(remaining_nulls.nlargest(10))
        
        X = X.fillna(X.median())
        print("Filled with column medians")
    
    print(f"\n" + "="*70)
    print(f"FINAL FEATURE SUMMARY")
    print(f"="*70)
    print(f"Total features: {X.shape[1]}")
    print(f"  - Historical lag features: {len([c for c in hist_cols if c in X.columns])}")
    print(f"  - Lag availability indicators: {len([c for c in X.columns if 'has_hist_' in c])}")
    print(f"  - Storm maturity features: 3")
    print(f"  - Other features: {X.shape[1] - len([c for c in hist_cols if c in X.columns]) - len([c for c in X.columns if 'has_hist_' in c]) - 3}")
    print(f"Total samples: {X.shape[0]}")
    print("="*70 + "\n")
    
    return X, targetColumns, remaining_nulls
