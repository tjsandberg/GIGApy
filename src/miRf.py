import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
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

    #df.dtypes.to_csv(args.scratchDir + 'dtypes.csv', index=True)
    targetColumns = dfUsage["Feature"][(dfUsage["Usage"] == "target")]
    dropFeatures = dfUsage["Feature"][(dfUsage["Usage"] == "target") | (dfUsage["Usage"] == "ignore")]
    
    # Drop target features and other features to be ignored
    X = df.drop(columns=dropFeatures)
    
    # Ignore non-numeric input columns
    X = X.select_dtypes(include=[np.number])
        
    print(f"Final number of features: {X.shape[1]} after dropping non-numeric columns")
    print(f"Number of samples: {X.shape[0]}")    

    # Ignore non-numeric input columns
    X = X.select_dtypes(include=[np.number])

    print(f"Final number of features: {X.shape[1]} after dropping non-numeric columns")
    print(f"Number of samples: {X.shape[0]}")
    print("\n" + "=" * 80)
    print("Use Mutual Information to order features by Importance for predicting the target.")
    print("Then use Random Forest to select the optimal number of features")
    print("=" * 80)

    # Do each target column
    for tc in targetColumns:
        print("\n" + "=" * 70)
        print(f"Processing {tc}")
        print("=" * 70)
        
        y = df[tc] # Do MI regression on full data set
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_importances = pd.DataFrame({
            'feature': X.columns,        
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        print(f"\nTop Features by Mutual Information")
        print(mi_importances.head(20))
        
        # ========== FEATURE SELECTION EXPERIMENT ==========
        print(f"\nFEATURE SELECTION EXPERIMENT FOR {tc}")
        
        # Test model performance with different numbers of top features
        feature_counts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200, 300, 400, X.shape[1]]
        #feature_counts = [5, 10]
        results = []
        best_test_r2 = 0
        best_n_features = 0
    
        # Split data set for Random Forest Experiments
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        for n_features in feature_counts:
            if n_features > X.shape[1]:
                n_features = X.shape[1]
            
            # Use top N features from combined ranking
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
            
            results.append({
                'n_features': n_features,
                'train_r2': train_r2,
                'test_r2': test_r2
            })

            print(f"Features: {n_features:3d} | Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")

        out_header_lines = f"RESULT: Optimal number of features is "
        out_header_lines += f"{best_n_features} which produces R2 = {best_test_r2:.4f}.\n"
        print(out_header_lines)
        out_header_lines += f"Features by Mutual Information for {tc} from input file: "
        out_header_lines += f"{args.dbaseInFile}.\n"

        # Create notes to save to output file
        notesData = {
            "Field": ["Target", "Input File", "Usage File", "OptimalNF", "R2"],
            "Value": [tc, args.dbaseInFile, args.dbUsage, best_n_features, best_test_r2]
        }
        notes = pd.DataFrame(notesData)
        
        # Create unique file name and write to it
        now = datetime.now()
        outFileName = f"{args.scratchDir}{tc}_{now:%Y%m%d%H%M%S}_featuresByMi.ods"
        with pd.ExcelWriter(outFileName) as writer:  
            mi_importances.to_excel(writer, sheet_name='MI_Imp')
            dfUsage.to_excel(writer, sheet_name='Usage')
            notes.to_excel(writer, sheet_name='Notes')
        print(f"{tc} output saved to '{outFileName}'")

    print("\n" + "=" * 50)
    print(f"EXPERIMENT COMPLETE")
    print("=" * 50)

except Exception as e:
    print(f"\nAn error occurred: {type(e).__name__}")
    print(f"Details: {e}")
    import traceback
    traceback.print_exc()
