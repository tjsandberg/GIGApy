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

# 1. Create argument parser object
parser = argparse.ArgumentParser(description="Use Mutual Information to order features by Importance for predicting the target. Then use Random Forest to select the optimal number of features.")

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
    
    dropFeatures = dfUsage["Feature"][(dfUsage["Usage"] == "ignore") | (dfUsage["Usage"] == "target")]
    #print(f"Dropped features:\n{dropFeatures}\n")
    # Drop target features and other features to be ignored
    X = df.drop(columns=dropFeatures)

    targetColumns = dfUsage["Feature"][(dfUsage["Usage"] == "target")]
    #print(f"Correlation Targets:\n{targetColumns}\n")
    
    # Ignore non-numeric input columns
    X = X.select_dtypes(include=[np.number])
    
    #  Use mean for missing values - TJS probably should do something better here when NULLs are not pre-removed.
    #X = X.fillna(X.mean())
    #y = y.fillna(y.mean())
    
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
        
        y = df[tc] # TJS fix this
    
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        #mi_scores = mutual_info_regression(X_train, y_train, random_state=42) # TJS seems not right
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_importances = pd.DataFrame({
            #'feature': X_train.columns,           # TJS seems not right
            'feature': X.columns,        
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        print(f"\nTop Features by Mutual Information")
        print(mi_importances.head(20))
        
        # ========== FEATURE SELECTION EXPERIMENT ==========
        print(f"\nFEATURE SELECTION EXPERIMENT FOR {tc}")
        
        # Test model performance with different numbers of top features
        #feature_counts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200, 300, 400, X.shape[1]]
        feature_counts = [5, 10]
        results = []
        best_test_r2 = 0
        best_n_features = 0

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
        
        # Save to CSV for further analysis
        out_header_lines = f"RESULT: Optimal number of features is "
        out_header_lines += f"{best_n_features} which produces R2 = {best_test_r2:.4f}.\n"
        print(out_header_lines)
        out_header_lines += f"Features by Mutual Information for {tc} from input file: "
        out_header_lines += f"{args.dbaseInFile}.\n"
        outFileName = args.scratchDir + tc + '_featuresByMi.csv'
        with open(outFileName, 'w') as f:
            f.write(out_header_lines.strip() + '\n') # .strip() removes leading/trailing white space
            mi_importances.to_csv(f, index=False) 
        print(f"All results saved to {outFileName}.\n")

    print("\n" + "=" * 50)
    print(f"EXPERIMENT COMPLETE")
    print("=" * 50)

except Exception as e:
    print(f"\nAn error occurred: {type(e).__name__}")
    print(f"Details: {e}")
    import traceback
    traceback.print_exc()
