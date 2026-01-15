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


    # ========== CONFIGURE CUSTOM SPLIT ==========
    # Specify which feature and threshold to use for the first split
    SPLIT_FEATURE = 'T2M_300_270_10'  # Change this to your desired feature name
    SPLIT_THRESHOLD = X_nonzero[SPLIT_FEATURE].median()


    print(f"\nForcing first split on: {SPLIT_FEATURE} <= {SPLIT_THRESHOLD}")

    # ========== CREATE MANUAL FIRST SPLIT ==========

    # Split training data based on your custom rule
    mask_train_LT = X_nonzero[SPLIT_FEATURE] <= SPLIT_THRESHOLD
    mask_train_GT = X_nonzero[SPLIT_FEATURE] > SPLIT_THRESHOLD

    X_LT = X_nonzero[mask_train_LT]
    X_GT = X_nonzero[mask_train_GT]

    print(f"\nLT branch (<=): {len(X_LT)} samples")
    print(f"GT branch (>): {len(X_GT)} samples")

    # ========== CORRELATE SUBTREES ==========
    print("\n" + "=" * 50)
    print("CORRELATING SUBTREES")
    print("=" * 50)
     
# Do each target column
    for tc in targetColumns:
        y = df[tc]
        y_LT = y[mask_train_LT]
        y_GT = y[mask_train_GT]        
        correlations_all = X_nonzero.corrwith(y).abs().sort_values(ascending=False)
        correlations_LT = X_LT.corrwith(y_LT).abs().sort_values(ascending=False)
        correlations_GT = X_GT.corrwith(y_GT).abs().sort_values(ascending=False)
        
        corr_importances_all = pd.DataFrame({
            'feature': correlations_all.index,
            'abs_corr_all': correlations_all.values
        })
        corr_importances_LT = pd.DataFrame({
            'feature': correlations_LT.index,
            'abs_corr_LT': correlations_LT.values
        })
        corr_importances_GT = pd.DataFrame({
            'feature': correlations_GT.index,
            'abs_corr_GT': correlations_GT.values
        })
        corr_importances = pd.merge(corr_importances_all, corr_importances_LT, on="feature")        
        corr_importances = pd.merge(corr_importances, corr_importances_GT, on="feature")
        
        '''
        print(f"\nTop 20 Features for {tc} by Absolute Correlation (LT):")
        print(corr_importances_LT.head(20))
        print(f"\nTop 20 Features for {tc} by Absolute Correlation (GT):")
        print(corr_importances_GT.head(20))
        print(f"\nTop 20 Features for {tc} by Absolute Correlation (all):")
        print(corr_importances.head(20))
        '''                
        # Create notes to save to output file
        notesData = {
            "Field": ["Target", "Input File", "Usage File", "SplitFeature", "SplitThresh"],
            "Value": [tc, args.dbaseInFile, args.dbUsage, SPLIT_FEATURE, str(SPLIT_THRESHOLD)]
        }
        notes = pd.DataFrame(notesData)
        
        # Create unique file name and write to it
        now = datetime.now()
        outFileName = f"{args.scratchDir}{tc}_{now:%Y%m%d%H%M%S}_corr.ods"
        with pd.ExcelWriter(outFileName) as writer:  
            corr_importances.to_excel(writer, sheet_name='Corr')
            dfUsage.to_excel(writer, sheet_name='Usage')
            notes.to_excel(writer, sheet_name='Notes')
        print(f"{tc} correlations saved to '{outFileName}'")

except Exception as e:
    print(f"\nAn error occurred: {type(e).__name__}")
    print(f"Details: {e}")
    import traceback
    traceback.print_exc()
