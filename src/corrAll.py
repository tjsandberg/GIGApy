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

parser = argparse.ArgumentParser(description="Compute all correlations for the input data")
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
    X, targetColumns = prepare_features(df, dfUsage, include_targets=True)
    
    # Compute all correlations
    correlations = X.corr().abs()

    # Create notes
    notes = create_notes_dataframe({
        "Input File": args.dbaseInFile,
        "Usage File": args.dbUsage
    })
    
    # Save results
    outFileName = generate_output_filename(args.scratchDir, "allCorr")
    save_results_to_excel(outFileName, {
        'Corr': correlations,
        'Usage': dfUsage,
        'Notes': notes
    })

except Exception as e:
    print(f"\nAn error occurred: {type(e).__name__}")
    print(f"Details: {e}")
    import traceback
    traceback.print_exc()
