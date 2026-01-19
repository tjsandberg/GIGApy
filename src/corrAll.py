import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# 1. Create argument parser object
parser = argparse.ArgumentParser(description="A script that computes correlations after splitting the data set around the mean/median value for a specific input.")

# 2. Add arguments
parser.add_argument('dbaseInFile', type=str, help="Path and file name for database input file in csv format.")
parser.add_argument('--dbUsage', type=str, help="Path and file name for database usage file in csv format.")
parser.add_argument('--scratchDir', type=str, default= './tmp/', help="(Optional) Path to directory for output files. Default: --scratchDir ./tmp/")

# 3. Parse the arguments from the command line
args = parser.parse_args()

try:

    # ========== LOAD Feature Usage FROM CSV ==========
    print(f"\nLOADING Feature Usage FROM {args.dbUsage}")
    dfUsage = pd.read_excel(args.dbUsage)
    feature_dtypes = dict(zip(dfUsage["Feature"], dfUsage["dtype"]))

    # Replace Int64/Int32/etc with float64/float32 for reading (floats can handle NaN)
    dtype_for_reading = {}
    int_to_float_map = {'Int64': 'float64', 'Int32': 'float32', 'Int16': 'float32', 'Int8': 'float32'}

    for col, dtype in feature_dtypes.items():
        if dtype in int_to_float_map:
            dtype_for_reading[col] = int_to_float_map[dtype]
        else:
            dtype_for_reading[col] = dtype
    #print(f"dtype_for_reading: \n{dtype_for_reading}")

    # ========== LOAD DATA FROM CSV ==========
    # Read with compatible dtypes
    print(f"\nLOADING DATA FROM {args.dbaseInFile}")
    df = pd.read_csv(args.dbaseInFile, sep=',', dtype=dtype_for_reading)
    print(f"Dataset shape: {df.shape}")

    # Convert float back to nullable Int after reading
    for col, dtype in feature_dtypes.items():
        if col in df.columns and dtype.startswith('Int'):
            df[col] = df[col].astype(dtype)
   
    dropFeatures = dfUsage["Feature"][(dfUsage["Usage"] == "ignore")]
    #print(f"Dropped features:\n{dropFeatures}\n")
    
    # Drop target features and other features to be ignored
    X = df.drop(columns=dropFeatures)
    
    # Ignore non-numeric input columns
    X = X.select_dtypes(include=[np.number])
        
    print(f"Final number of features: {X.shape[1]} after dropping non-numeric columns")
    print(f"Number of samples: {X.shape[0]}")    
    correlations = X.corr().abs()

    # Create notes to save to output file
    notesData = {
        "Field": ["Input File", "Usage File"],
        "Value": [args.dbaseInFile, args.dbUsage]
    }
    notes = pd.DataFrame(notesData)
    
    # Create unique file name and write to it
    now = datetime.now()
    outFileName = f"{args.scratchDir}allCorr_{now:%Y%m%d%H%M%S}.ods"
    with pd.ExcelWriter(outFileName) as writer:  
        correlations.to_excel(writer, sheet_name='Corr')
        dfUsage.to_excel(writer, sheet_name='Usage')
        notes.to_excel(writer, sheet_name='Notes')
    print(f"All correlations saved to '{outFileName}'")

except Exception as e:
    print(f"\nAn error occurred: {type(e).__name__}")
    print(f"Details: {e}")
    import traceback
    traceback.print_exc()
