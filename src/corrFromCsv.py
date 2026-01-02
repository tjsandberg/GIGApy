import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# 1. Create argument parser object
parser = argparse.ArgumentParser(description="A script that computes correlations. TJS add more here.")

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

    targetColumns = dfUsage["Feature"][(dfUsage["Usage"] == "target")]
    #print(f"Correlation Targets:\n{targetColumns}\n")
    
    dropFeatures = dfUsage["Feature"][(dfUsage["Usage"] == "target") | (dfUsage["Usage"] == "ignore")]
    #print(f"Dropped features:\n{dropFeatures}\n")
    
    # Drop target features and other features to be ignored
    X = df.drop(columns=dropFeatures)
    
    # Ignore non-numeric input columns
    X = X.select_dtypes(include=[np.number])
        
    print(f"Final number of features: {X.shape[1]} after dropping non-numeric columns")
    print(f"Number of samples: {X.shape[0]}")    
    
    # ========== CORRELATION WITH TARGET ==========
    print("\n" + "=" * 50)
    print("CORRELATION WITH TARGET")
    print("(Simple linear relationship)")
    print("=" * 50)
    
    # Check for zero-variance features
    feature_std = X.std()
    zero_var_features = feature_std[feature_std == 0].index.tolist()
    
    if zero_var_features:
        print(f"\nWarning: {len(zero_var_features)} features have zero variance (constant values)")
        print(f"These features will be excluded from correlation analysis")
        print(f"Zero-variance features: {zero_var_features[:10]}{'...' if len(zero_var_features) > 10 else ''}")
        
        # Remove zero-variance features for correlation calculation
        X_nonzero = X.drop(columns=zero_var_features)
    else:
        X_nonzero = X
     
# Do first target column
    for tc in targetColumns:
        y = df[tc]
        correlations = X_nonzero.corrwith(y).abs().sort_values(ascending=False)
        
        # Add back zero-variance features with correlation = 0
        for feat in zero_var_features:
            correlations[feat] = 0.0
        
        correlations = correlations.sort_values(ascending=False)
        
        corr_importances = pd.DataFrame({
            'feature': correlations.index,
            'abs_corr': correlations.values
        })
        
        print(f"\nTop 20 Features for {tc} by Absolute Correlation:")
        print(corr_importances.head(20))
        
        # Save to CSV for further analysis
        out_header_lines = 'Correlations for ' + tc + ' from input file: ' + args.dbaseInFile + '.'
        outFileName = args.scratchDir + tc + '_correlations.csv'
        with open(outFileName, 'w') as f:
            f.write(out_header_lines.strip() + '\n') # .strip() removes leading/trailing white space for clean output
            corr_importances.to_csv(f, index=False) 
        print(f"{tc} correlations saved to '{outFileName}'")

except Exception as e:
    print(f"\nAn error occurred: {type(e).__name__}")
    print(f"Details: {e}")
    import traceback
    traceback.print_exc()
