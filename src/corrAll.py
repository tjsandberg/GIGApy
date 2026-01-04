import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# 1. Create argument parser object
parser = argparse.ArgumentParser(description="A script that computes correlations for all columns in input csv.")

# 2. Add arguments
parser.add_argument('dbaseInFile', type=str, help="Path and file name for database input file in csv format.")
parser.add_argument('--dbUsage', type=str, help="Path and file name for database usage file in csv format.")
parser.add_argument('--scratchDir', type=str, default= './tmp/', help="(Optional) Path to directory for output files. Default: --scratchDir ./tmp/")

# 3. Parse the arguments from the command line
args = parser.parse_args()

try:

    # ========== LOAD DATA FROM CSV ==========
    print(f"\nLOADING DATA FROM {args.dbaseInFile}")
    df = pd.read_csv(args.dbaseInFile, sep=',')
    print(f"Dataset shape: {df.shape}")

# ========== LOAD Feature Usage FROM CSV ==========
    print(f"\nLOADING Feature Usage FROM {args.dbUsage}")
    dfUsage = pd.read_csv(args.dbUsage, sep=',')

    targetColumns = dfUsage["Feature"][(dfUsage["Usage"] == "target") | (dfUsage["Usage"] == "use")]
    #print(f"Correlation Targets:\n{targetColumns}\n")
    
    dropFeatures = dfUsage["Feature"][(dfUsage["Usage"] == "ignore")]
    #print(f"Dropped features:\n{dropFeatures}\n")
    
    # Drop target features and other features to be ignored
    X = df.drop(columns=dropFeatures)
    
    # Ignore non-numeric input columns
    X = X.select_dtypes(include=[np.number])
        
    print(f"Final number of features: {X.shape[1]} after dropping non-numeric columns")
    print(f"Number of samples: {X.shape[0]}")    
    correlations = X.corr().abs()

    # Save to CSV for further analysis
    out_header_lines = 'Correlations for ' + ' from input file: ' + args.dbaseInFile + '.'
    outFileName = args.scratchDir + 'allCorrelations.csv'
    with open(outFileName, 'w') as f:
        f.write(out_header_lines.strip() + '\n') # .strip() removes leading/trailing white space for clean output
        correlations.to_csv(f, index=False) 
    print(f"correlations saved to '{outFileName}'")

except Exception as e:
    print(f"\nAn error occurred: {type(e).__name__}")
    print(f"Details: {e}")
    import traceback
    traceback.print_exc()
