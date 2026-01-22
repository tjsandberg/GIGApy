import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
import argparse

# Import shared utilities
from data_io import (
    load_database_with_dtypes,
    prepare_features,
    save_results_to_excel,
    create_notes_dataframe,
    generate_output_filename,
    calculate_data_quality_stats
)

# Create argument parser
parser = argparse.ArgumentParser(description="Report Data Quality statistcs.")
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
    
    # Add to notes when saving results:
    quality_stats = calculate_data_quality_stats(df, X)

    stats = create_notes_dataframe({
        #"Target": tc,
        #"OptimalNF": best_n_features,
        #"R2": best_test_r2,
        **quality_stats  # Unpack quality stats into notes
    })
    print(f"Stats: \n{stats.to_string(index=False)}")
    
    notes = create_notes_dataframe({
        "Input File": args.dbaseInFile,
        "Usage File": args.dbUsage
    })
    
    outFileName = generate_output_filename(args.scratchDir, f"_dq")
    save_results_to_excel(outFileName, {
        'Stats': stats,
        'Usage': dfUsage,
        'Notes': notes
    })

except Exception as e:
    print(f"\nAn error occurred: {type(e).__name__}")
    print(f"Details: {e}")
    import traceback
    traceback.print_exc()
