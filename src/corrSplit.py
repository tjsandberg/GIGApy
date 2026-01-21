import numpy as np
import pandas as pd
import argparse

from data_io import (
    load_database_with_dtypes,
    prepare_features,
    save_results_to_excel,
    create_notes_dataframe,
    generate_output_filename
)

parser = argparse.ArgumentParser(
    description="Compute correlations after splitting the data set around "
                "the mean/median value for a specific input."
)
parser.add_argument('dbaseInFile', type=str, 
                   help="Path and file name for database input file in csv format.")
parser.add_argument('--dbUsage', type=str, 
                   help="Path and file name for database usage file in csv format.")
parser.add_argument('--scratchDir', type=str, default='./tmp/', 
                   help="(Optional) Path to directory for output files. Default: ./tmp/")

args = parser.parse_args()

try:
    # Load data
    df, dfUsage = load_database_with_dtypes(args.dbaseInFile, args.dbUsage)
    X, targetColumns = prepare_features(df, dfUsage, include_targets=False)
    
    # Check for zero-variance features
    print("\n" + "=" * 50)
    print("CHECKING FOR ZERO-VARIANCE FEATURES")
    print("=" * 50)
    
    feature_std = X.std()
    zero_var_features = feature_std[feature_std == 0].index.tolist()
    
    if zero_var_features:
        print(f"\nWarning: {len(zero_var_features)} features have zero variance")
        print(f"These features will be excluded from correlation analysis")
        X_nonzero = X.drop(columns=zero_var_features)
    else:
        X_nonzero = X
        print("No zero-variance features found")

    # Configure custom split
    SPLIT_FEATURE = 'T2M_300_270_10'
    SPLIT_THRESHOLD = X_nonzero[SPLIT_FEATURE].median()
    
    print(f"\nForcing split on: {SPLIT_FEATURE} <= {SPLIT_THRESHOLD}")

    # Split data
    mask_LT = X_nonzero[SPLIT_FEATURE] <= SPLIT_THRESHOLD
    mask_GT = X_nonzero[SPLIT_FEATURE] > SPLIT_THRESHOLD

    X_LT = X_nonzero[mask_LT]
    X_GT = X_nonzero[mask_GT]

    print(f"LT branch (<=): {len(X_LT)} samples")
    print(f"GT branch (>): {len(X_GT)} samples")

    # Process each target column
    print("\n" + "=" * 50)
    print("CORRELATING SUBTREES")
    print("=" * 50)
    
    for tc in targetColumns:
        y = df[tc]
        y_LT = y[mask_LT]
        y_GT = y[mask_GT]
        
        # Calculate correlations
        correlations_all = X_nonzero.corrwith(y).abs().sort_values(ascending=False)
        correlations_LT = X_LT.corrwith(y_LT).abs().sort_values(ascending=False)
        correlations_GT = X_GT.corrwith(y_GT).abs().sort_values(ascending=False)

        # Create DataFrames with ranks
        corr_all = pd.DataFrame({
            'feature': correlations_all.index,
            'abs_corr_all': correlations_all.values,
            'ALL_rank': range(1, len(correlations_all) + 1)
        })
        
        corr_LT = pd.DataFrame({
            'feature': correlations_LT.index,
            'abs_corr_LT': correlations_LT.values,
            'LT_rank': range(1, len(correlations_LT) + 1)
        })
        
        corr_GT = pd.DataFrame({
            'feature': correlations_GT.index,
            'abs_corr_GT': correlations_GT.values,
            'GT_rank': range(1, len(correlations_GT) + 1)
        })
        
        # Merge all correlation results
        corr_importances = pd.merge(corr_all, corr_LT, on="feature")
        corr_importances = pd.merge(corr_importances, corr_GT, on="feature")
        
        print(f"\nCorrelations for {tc}:")
        print(corr_importances.head(20))

        # Create notes
        notes = create_notes_dataframe({
            "Target": tc,
            "Input File": args.dbaseInFile,
            "Usage File": args.dbUsage,
            "SplitFeature": SPLIT_FEATURE,
            "SplitThresh": SPLIT_THRESHOLD
        })
        
        # Save results
        outFileName = generate_output_filename(args.scratchDir, f"{tc}_corr")
        save_results_to_excel(outFileName, {
            'Corr': corr_importances,
            'Usage': dfUsage,
            'Notes': notes
        })

except Exception as e:
    print(f"\nAn error occurred: {type(e).__name__}")
    print(f"Details: {e}")
    import traceback
    traceback.print_exc()
