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
    prepare_hurricane_features_with_lags,  # <-- Use this instead of prepare_features
    prepare_hurricane_features_simplified,
    save_results_to_excel,
    create_notes_dataframe,
    generate_output_filename
)

# Create argument parser
parser = argparse.ArgumentParser(
    description="Use Mutual Information to order features by importance, "
                "then Random Forest to select optimal number of features."
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
    #df = df[df["PredLat_24"].notna()] # Remove tests with no output value
    #X, targetColumns, remainingNulls = prepare_hurricane_features_with_lags(df, dfUsage, include_targets=False, saveTLMdb=False)
    X, targetColumns = prepare_hurricane_features_simplified(df, dfUsage, include_targets=False)
    #X, targetColumns = prepare_features(df, dfUsage, include_targets=False)

    print("\n" + "=" * 80)
    print("Use Mutual Information to order features by Importance for predicting the target.")
    print("Then use Random Forest to select the optimal number of features")
    print("=" * 80)

    # Process each target column
    for tc in targetColumns:
        print("\n" + "=" * 70)
        print(f"Processing {tc}")
        print("=" * 70)
        
        y = df[tc]
        
        # Calculate Mutual Information on full dataset
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_importances = pd.DataFrame({
            'feature': X.columns,        
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        print(f"\nTop Features by Mutual Information")
        print(mi_importances.head(20))
        
        # Feature selection experiment
        print(f"\nFEATURE SELECTION EXPERIMENT FOR {tc}")
        
        feature_counts = [5, 10, 15, 20, 30, 45, 75, 200, 400, X.shape[1]]
        best_test_r2 = 0
        best_n_features = 0
    
        # Split data for Random Forest experiments
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        for n_features in feature_counts:
            if n_features > X.shape[1]:
                n_features = X.shape[1]
            
            # Use top N features
            top_features = mi_importances.head(n_features)['feature'].values
            X_train_subset = X_train[top_features]
            X_test_subset = X_test[top_features]
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                         random_state=42, n_jobs=-1)
            model.fit(X_train_subset, y_train)
            
            # Evaluate
            train_pred = model.predict(X_train_subset)
            test_pred = model.predict(X_test_subset)
            
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            if test_r2 > best_test_r2:
                best_test_r2 = test_r2
                best_n_features = n_features

            print(f"Features: {n_features:3d} | Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")

        print(f"\nRESULT: Optimal number of features is {best_n_features} "
              f"which produces R² = {best_test_r2:.4f}")

        # Create notes
        notes = create_notes_dataframe({
            "Target": tc,
            "Input File": args.dbaseInFile,
            "Usage File": args.dbUsage,
            "OptimalNF": best_n_features,
            "R2": best_test_r2
        })
        
        # Save results
        outFileName = generate_output_filename(args.scratchDir, f"{tc}_featuresByMi")
        save_results_to_excel(outFileName, {
            'MI_Imp': mi_importances,
            'Usage': dfUsage,
            #'Nulls': remainingNulls,
            'Notes': notes
        })

    print("\n" + "=" * 50)
    print("EXPERIMENT COMPLETE")
    print("=" * 50)

except Exception as e:
    print(f"\nAn error occurred: {type(e).__name__}")
    print(f"Details: {e}")
    import traceback
    traceback.print_exc()
