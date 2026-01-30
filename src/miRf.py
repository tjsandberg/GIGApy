import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import mutual_info_regression
import argparse

# Import shared utilities
from data_io import (
    load_database_with_dtypes,
    prepare_hurricane_features_simplified,
    save_results_to_excel,
    create_notes_dataframe,
    generate_output_filename
)

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great circle distance between points in nautical miles.
    
    Parameters:
    -----------
    lat1, lon1, lat2, lon2 : array-like
        Coordinates in decimal degrees
        
    Returns:
    --------
    distance : array-like
        Distance in nautical miles
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of Earth in nautical miles (1 nm = 1.852 km, Earth radius ~ 6371 km)
    r_nm = 6371 / 1.852
    
    return r_nm * c

def calculate_position_rmse(y_true_lat, y_true_lon, y_pred_lat, y_pred_lon):
    """
    Calculate RMSE in nautical miles for position predictions.
    
    Parameters:
    -----------
    y_true_lat, y_true_lon : array-like
        True positions
    y_pred_lat, y_pred_lon : array-like
        Predicted positions
        
    Returns:
    --------
    rmse_nm : float
        Root mean square error in nautical miles
    mean_error_nm : float
        Mean error in nautical miles
    """
    distances = haversine_distance(y_true_lat, y_true_lon, y_pred_lat, y_pred_lon)
    rmse_nm = np.sqrt(np.mean(distances**2))
    mean_error_nm = np.mean(distances)
    return rmse_nm, mean_error_nm

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
    X, targetColumns, null_count, remaining_nulls_df = prepare_hurricane_features_simplified(df, dfUsage, include_targets=False)

    print("\n" + "=" * 80)
    print("Use Mutual Information to order features by Importance for predicting the target.")
    print("Then use Random Forest to select the optimal number of features")
    print("=" * 80)

    # Storage for predictions to calculate combined RMSE
    predictions_dict = {}

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
        best_test_r2 = -np.inf
        best_n_features = 0
        best_test_pred = None
        results = []
        
        # Split data for Random Forest experiments (SAME random_state for all targets!)
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
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            if test_r2 > best_test_r2:
                best_test_r2 = test_r2
                best_n_features = n_features
                best_test_pred = test_pred
            
            # Approximate RMSE in nautical miles (1 degree ≈ 60 nm)
            test_rmse_nm = test_rmse * 60
            
            print(f"Features: {n_features:3d} | Train R²: {train_r2:.4f} | "
                  f"Test R²: {test_r2:.4f} | RMSE: {test_rmse_nm:.1f} nm")

            results.append({
                'n_features': n_features,
                'train_r2': f"{train_r2:.4f}",
                'test_r2': f"{test_r2:.4f}",
                'test_rmse_nm': f"{test_rmse_nm:.1f}"
            })

        print(f"\nRESULT: Optimal number of features is {best_n_features} "
              f"which produces R² = {best_test_r2:.4f}")

        # Store predictions for combined RMSE calculation
        predictions_dict[tc] = {
            'y_test': y_test,
            'y_pred': best_test_pred,
            'best_n_features': best_n_features,
            'best_r2': best_test_r2,
            'test_indices': y_test.index
        }

        # Create notes
        notes = create_notes_dataframe({
            "Target": tc,
            "Input File": args.dbaseInFile,
            "Usage File": args.dbUsage,
            "Hist Nulls -> 0": null_count,
            "Rem Nulls -> median": remaining_nulls_df['Null_Count'].sum(),
            "OptimalNF": best_n_features,
            "R2": f"{best_test_r2:.4f}"
        })
        
        # Save results
        results_df = pd.DataFrame(results)
        outFileName = generate_output_filename(args.scratchDir, f"{tc}_featuresByMi")
        save_results_to_excel(outFileName, {
            'MI_Imp': mi_importances,
            'Usage': dfUsage,
            'Nulls': remaining_nulls_df,
            'RfResults': results_df,
            'Notes': notes
        })

    # Calculate combined position RMSE if both lat and lon were processed
    if 'PredLat_24' in predictions_dict and 'PredLon_24' in predictions_dict:
        print("\n" + "=" * 70)
        print("COMBINED POSITION PREDICTION METRICS")
        print("=" * 70)
        
        lat_pred = predictions_dict['PredLat_24']
        lon_pred = predictions_dict['PredLon_24']
        
        # Get test set indices (should be the same for both)
        test_idx = lat_pred['test_indices']
        
        # Calculate position RMSE using haversine distance
        rmse_nm, mean_error_nm = calculate_position_rmse(
            lat_pred['y_test'].values,
            lon_pred['y_test'].values,
            lat_pred['y_pred'],
            lon_pred['y_pred']
        )
        
        # Calculate individual prediction errors for statistics
        pred_distances = haversine_distance(
            lat_pred['y_test'].values,
            lon_pred['y_test'].values,
            lat_pred['y_pred'],
            lon_pred['y_pred']
        )
        
        print(f"\n24-Hour Position Prediction Performance:")
        print(f"  Position RMSE: {rmse_nm:.2f} nautical miles")
        print(f"  Mean error: {mean_error_nm:.2f} nm")
        print(f"  Median error: {np.median(pred_distances):.2f} nm")
        print(f"  95th percentile: {np.percentile(pred_distances, 95):.2f} nm")
        print(f"  Max error: {np.max(pred_distances):.2f} nm")
        
        print(f"\nIndividual Component Performance:")
        print(f"  Latitude R²: {lat_pred['best_r2']:.4f} (optimal features: {lat_pred['best_n_features']})")
        print(f"  Longitude R²: {lon_pred['best_r2']:.4f} (optimal features: {lon_pred['best_n_features']})")
        
        # Save combined metrics
        combined_notes = create_notes_dataframe({
            "Prediction": "24-hour Position",
            "Input File": args.dbaseInFile,
            "Usage File": args.dbUsage,
            "Position RMSE (nm)": f"{rmse_nm:.2f}",
            "Mean Error (nm)": f"{mean_error_nm:.2f}",
            "Median Error (nm)": f"{np.median(pred_distances):.2f}",
            "95th Percentile (nm)": f"{np.percentile(pred_distances, 95):.2f}",
            "Lat Optimal Features": lat_pred['best_n_features'],
            "Lat R²": f"{lat_pred['best_r2']:.4f}",
            "Lon Optimal Features": lon_pred['best_n_features'],
            "Lon R²": f"{lon_pred['best_r2']:.4f}"
        })
        
        outFileName = generate_output_filename(args.scratchDir, "Combined_Position_Metrics")
        save_results_to_excel(outFileName, {
            'Metrics': combined_notes
        })
        
        print(f"\nCombined metrics saved to '{outFileName}'")

    print("\n" + "=" * 50)
    print("EXPERIMENT COMPLETE")
    print("=" * 50)

except Exception as e:
    print(f"\nAn error occurred: {type(e).__name__}")
    print(f"Details: {e}")
    import traceback
    traceback.print_exc()
