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
            df.to_excel(writer, sheet_name=sheet_name)
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
