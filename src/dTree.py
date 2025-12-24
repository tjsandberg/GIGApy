import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns

# ========== LOAD DATA FROM CSV ==========
scratchDir = './tmp/'
dataDir = '/home/tom/Personal/fun/Giga/'
csv_file = dataDir + 'DataDictionary2u8noNull.csv'
target_column = 'PredLon_24'

print(f"\nLOADING DATA FROM {csv_file}")
print(f"Predicting {target_column}")

try:
    df = pd.read_csv(csv_file, sep=',')
    
    print(f"Dataset shape: {df.shape}")
    # print(f"Number of features: {df.shape[1] - 5}")  # Minus target and TestId
    
    # Extract features and target
    X = df.drop(columns=['Test', 'Name', 'Desig', 'StormNum', 'SampleNum', 'QuadSample',
        'PredLat_24', 'PredLon_24', 'PredBearing_24', 'PredDistance_24',
        'HistBearing_6', 'HistDistance_6', 'HistBearing_12', 'HistDistance_12', 'HistBearing_18', 'HistDistance_18',
        'HistBearing_24', 'HistDistance_24', 'HistBearing_30', 'HistDistance_30', 'HistBearing_36', 'HistDistance_36',
        'HistBearing_42', 'HistDistance_42', 'HistBearing_48', 'HistDistance_48', 'HistBearing_54', 'HistDistance_54',
        'HistBearing_60', 'HistDistance_60', 'HistBearing_66', 'HistDistance_66', 'HistBearing_72', 'HistDistance_72'
        ])
     
    y = df[target_column]
    
    # Ignore non-numeric input columns
    X = X.select_dtypes(include=[np.number])
    
    #  Use mean for missing values - TJS probably should do something better here when NULLs are not pre-removed.
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    print(f"Final number of features: {X.shape[1]} after dropping non-numeric columns")
    print(f"Number of samples: {X.shape[0]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # ========== METHOD 1: DECISION TREE FEATURE IMPORTANCE ==========
    print("\n" + "=" * 50)
    print("METHOD 1: DECISION TREE FEATURE IMPORTANCE")
    print("=" * 50)
    
    dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt_model.fit(X_train, y_train)
    
    dt_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Features by Decision Tree:")
    print(dt_importances.head(20))
    
    # ========== METHOD 2: RANDOM FOREST FEATURE IMPORTANCE ==========
    print("\n" + "=" * 50)
    print("METHOD 2: RANDOM FOREST FEATURE IMPORTANCE")
    print("(More robust than single tree)")
    print("=" * 50)
    
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                     random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    rf_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Features by Random Forest:")
    print(rf_importances.head(20))
    
    # Model performance
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    print(f"\nRandom Forest Test R²: {rf_r2:.4f}")
    
    # ========== METHOD 3: MUTUAL INFORMATION ==========
    print("\n" + "=" * 50)
    print("METHOD 3: MUTUAL INFORMATION")
    print("(Captures non-linear relationships)")
    print("=" * 50)
    
    mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
    mi_importances = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    print("\nTop 20 Features by Mutual Information:")
    print(mi_importances.head(20))
    
    # ========== METHOD 4: CORRELATION WITH TARGET ==========
    print("\n" + "=" * 50)
    print("METHOD 4: CORRELATION WITH TARGET")
    print("(Simple linear relationship)")
    print("=" * 50)
    
    # Check for zero-variance features
    feature_std = X_train.std()
    zero_var_features = feature_std[feature_std == 0].index.tolist()
    
    if zero_var_features:
        print(f"\nWarning: {len(zero_var_features)} features have zero variance (constant values)")
        print(f"These features will be excluded from correlation analysis")
        print(f"Zero-variance features: {zero_var_features[:10]}{'...' if len(zero_var_features) > 10 else ''}")
        
        # Remove zero-variance features for correlation calculation
        X_train_nonzero = X_train.drop(columns=zero_var_features)
    else:
        X_train_nonzero = X_train
    
    correlations = X_train_nonzero.corrwith(y_train).abs().sort_values(ascending=False)
    
    # Add back zero-variance features with correlation = 0
    for feat in zero_var_features:
        correlations[feat] = 0.0
    
    correlations = correlations.sort_values(ascending=False)
    
    corr_importances = pd.DataFrame({
        'feature': correlations.index,
        'abs_correlation': correlations.values
    })
    
    print("\nTop 20 Features by Absolute Correlation:")
    print(corr_importances.head(20))
    
    # ========== COMBINED RANKING ==========
    print("\n" + "=" * 50)
    print("COMBINED FEATURE RANKING")
    print("(Using rank-based scoring)")
    print("=" * 50)
    
    # Create rank-based scores (lower rank = more important)
    # Assign ranks where rank 1 = most important
    dt_ranked = dt_importances.copy()
    dt_ranked['dt_rank'] = range(1, len(dt_ranked) + 1)
    
    rf_ranked = rf_importances.copy()
    rf_ranked['rf_rank'] = range(1, len(rf_ranked) + 1)
    
    mi_ranked = mi_importances.copy()
    mi_ranked['mi_rank'] = range(1, len(mi_ranked) + 1)
    
    corr_ranked = corr_importances.copy()
    corr_ranked['corr_rank'] = range(1, len(corr_ranked) + 1)
    
    # Merge all methods with their original scores
    combined = dt_ranked[['feature', 'importance', 'dt_rank']].merge(
        rf_ranked[['feature', 'importance', 'rf_rank']], 
        on='feature', suffixes=('_dt', '_rf')
    )
    combined = combined.merge(mi_ranked[['feature', 'mi_score', 'mi_rank']], on='feature')
    combined = combined.merge(corr_ranked[['feature', 'abs_correlation', 'corr_rank']], on='feature')
    
    # Calculate average rank (lower is better)
    combined['avg_rank'] = (
        combined['dt_rank'] + 
        combined['rf_rank'] + 
        combined['mi_rank'] + 
        combined['corr_rank']
    ) / 4
    
    # Sort by average rank
    combined = combined.sort_values('avg_rank', ascending=True)
    
    # Add a normalized score (inverse of rank, scaled 0-1) for visualization
    combined['combined_score'] = 1 - (combined['avg_rank'] - 1) / (len(combined) - 1)
    
    print("\nTop 30 Features by Combined Ranking:")
    print(combined[['feature', 'avg_rank', 'importance_rf', 'mi_score', 'abs_correlation', 
                    'dt_rank', 'rf_rank', 'mi_rank', 'corr_rank']].head(30))
    
    # Save to CSV for further analysis
    combined.to_csv(scratchDir + 'feature_importances_combined.csv', index=False)
    print("\nFull feature importance rankings saved to 'feature_importances_combined.csv'")
    
    # ========== VISUALIZATIONS ==========
    print("\n" + "=" * 50)
    print("CREATING VISUALIZATIONS")
    print("=" * 50)
    
    # Plot 1: Top N features comparison across methods
    top_n = 30
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Decision Tree
    top_dt = dt_importances.head(top_n)
    axes[0, 0].barh(range(len(top_dt)), top_dt['importance'])
    axes[0, 0].set_yticks(range(len(top_dt)))
    axes[0, 0].set_yticklabels(top_dt['feature'], fontsize=8)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel('Importance')
    axes[0, 0].set_title('Decision Tree Feature Importance (Top 30)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Random Forest
    top_rf = rf_importances.head(top_n)
    axes[0, 1].barh(range(len(top_rf)), top_rf['importance'])
    axes[0, 1].set_yticks(range(len(top_rf)))
    axes[0, 1].set_yticklabels(top_rf['feature'], fontsize=8)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel('Importance')
    axes[0, 1].set_title('Random Forest Feature Importance (Top 30)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mutual Information
    top_mi = mi_importances.head(top_n)
    axes[1, 0].barh(range(len(top_mi)), top_mi['mi_score'])
    axes[1, 0].set_yticks(range(len(top_mi)))
    axes[1, 0].set_yticklabels(top_mi['feature'], fontsize=8)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel('MI Score')
    axes[1, 0].set_title('Mutual Information (Top 30)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation
    top_corr = corr_importances.head(top_n)
    axes[1, 1].barh(range(len(top_corr)), top_corr['abs_correlation'])
    axes[1, 1].set_yticks(range(len(top_corr)))
    axes[1, 1].set_yticklabels(top_corr['feature'], fontsize=8)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel('Absolute Correlation')
    axes[1, 1].set_title('Correlation with Target (Top 30)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(scratchDir + 'feature_importance_comparison.png', dpi=150, bbox_inches='tight')
    print("Feature importance comparison saved to 'feature_importance_comparison.png'")
    
    # Plot 2: Combined ranking
    fig, ax = plt.subplots(figsize=(12, 14))
    top_combined = combined.head(40)
    
    y_pos = np.arange(len(top_combined))
    ax.barh(y_pos, top_combined['combined_score'], alpha=0.7, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_combined['feature'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Combined Score (based on average rank)')
    ax.set_title('Top 40 Features by Combined Ranking')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(scratchDir + 'combined_feature_ranking.png', dpi=150, bbox_inches='tight')
    print("Combined feature ranking saved to 'combined_feature_ranking.png'")
    
    # Plot 3: Feature importance distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(dt_importances['importance'], bins=50, edgecolor='black')
    axes[0, 0].set_xlabel('Importance')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Decision Tree Importance Distribution')
    axes[0, 0].set_yscale('log')
    
    axes[0, 1].hist(rf_importances['importance'], bins=50, edgecolor='black')
    axes[0, 1].set_xlabel('Importance')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Random Forest Importance Distribution')
    axes[0, 1].set_yscale('log')
    
    axes[1, 0].hist(mi_importances['mi_score'], bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('MI Score')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Mutual Information Distribution')
    axes[1, 0].set_yscale('log')
    
    axes[1, 1].hist(corr_importances['abs_correlation'], bins=50, edgecolor='black')
    axes[1, 1].set_xlabel('Absolute Correlation')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Correlation Distribution')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(scratchDir + 'importance_distributions.png', dpi=150, bbox_inches='tight')
    print("Importance distributions saved to 'importance_distributions.png'")
    
    # ========== FEATURE SELECTION EXPERIMENT ==========
    print("\n" + "=" * 50)
    print("FEATURE SELECTION EXPERIMENT")
    print("=" * 50)
    
    # Test model performance with different numbers of top features
    feature_counts = [5, 10, 20, 30, 50, 100, X.shape[1]]
    results = []
    
    for n_features in feature_counts:
        if n_features > X.shape[1]:
            n_features = X.shape[1]
        
        # Use top N features from combined ranking
        top_features = combined.head(n_features)['feature'].values
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
        
        results.append({
            'n_features': n_features,
            'train_r2': train_r2,
            'test_r2': test_r2
        })
        
        print(f"Features: {n_features:3d} | Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    
    # Plot feature selection results
    results_df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df['n_features'], results_df['train_r2'], 
            marker='o', label='Training R²', linewidth=2)
    ax.plot(results_df['n_features'], results_df['test_r2'], 
            marker='s', label='Test R²', linewidth=2)
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('R² Score')
    ax.set_title('Model Performance vs Number of Features')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(scratchDir + 'feature_selection_performance.png', dpi=150, bbox_inches='tight')
    print("\nFeature selection performance saved to 'feature_selection_performance.png'")
    
    # ========== RECOMMENDATIONS ==========
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)
    
    # Find optimal number of features (where test R² plateaus)
    optimal_idx = results_df['test_r2'].idxmax()
    optimal_n = results_df.loc[optimal_idx, 'n_features']
    optimal_r2 = results_df.loc[optimal_idx, 'test_r2']
    
    print(f"\nOptimal number of features: {optimal_n}")
    print(f"Test R² with optimal features: {optimal_r2:.4f}")
    print(f"Test R² with all features: {results_df.iloc[-1]['test_r2']:.4f}")
    
    print("\nTop 20 recommended features to focus on:")
    print(combined[['feature', 'avg_rank', 'importance_rf', 'mi_score', 'abs_correlation']].head(20).to_string(index=False))
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE!")
    print("=" * 50)

except Exception as e:
    print(f"\nAn error occurred: {type(e).__name__}")
    print(f"Details: {e}")
    import traceback
    traceback.print_exc()
