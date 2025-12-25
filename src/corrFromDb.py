import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import getpass
import sys

#import pymysql
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Use 'Qt5Agg' or 'Qtagg' if you installed PyQt
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Config without the password
DB_CONFIG = {
    'user': 'gigadev',
    'host': 'giga-db-1-instance-1.c9kea8w2c1dc.us-east-2.rds.amazonaws.com', 
    'database': 'solar_system_3'
#    'host': '127.0.0.1',
#    'database': 'tGigaLocal',
#    'password': None # We will fill this in at runtime
}

# 2. Securely capture password (hidden on screen)
user_password = getpass.getpass(f"Enter password for {DB_CONFIG['user']}: ")

try:
    # 3. Robustly encode the password to handle characters like @, $, !, /
    # This prevents the "Name or service not known" error you saw earlier
    safe_password = urllib.parse.quote_plus(user_password)

    # 4. Construct the engine using the official mysql-connector driver
    # Format: mysql+mysqlconnector://user:password@host/database
    connection_url = (
        f"mysql+mysqlconnector://{DB_CONFIG['user']}:{safe_password}"
        f"@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    )
    
    engine = create_engine(connection_url)
    
    # Test the connection immediately
    with engine.connect() as conn:
        print("Successfully connected to the remote RDS instance.")

except Exception as e:
    print(f"Failed to connect: {e}")
    sys.exit(1)

# Now proceed with pd.read_sql_table(INPUT_TABLE, con=engine)...

INPUT_TABLE = 'widePrettyInputs'
OUTPUT_TABLE = 'widePrettyOutputs'
JOIN_COLUMN = 'Test'

# 3. Read data from MySQL into Pandas DataFrames
try:
    # Read input data (Input_1 to Input_10 + TestId)
    df_inputs = pd.read_sql_table(INPUT_TABLE, con=engine)
    # Read output data (Output column + TestId)
    df_outputs = pd.read_sql_table(OUTPUT_TABLE, con=engine)

    print(f"Successfully read data from {INPUT_TABLE} and {OUTPUT_TABLE}.")

except Exception as e:
    print(f"Error reading data from MySQL tables: {e}")
    sys.exit(1)

# 4. Merge the DataFrames on the common column (TestId)
# The merge function combines dataframes based on a common column
combined_df = pd.merge(df_inputs, df_outputs, on=JOIN_COLUMN, how='inner')

# Drop the TestId column before correlation analysis as it is an ID, not an input variable
analysis_df = combined_df.drop(columns=[JOIN_COLUMN])

# 4.5 Filter out constant columns (zero variance)
# A threshold of 0.0 means we strictly remove columns where all values are identical
constant_columns = analysis_df.columns[analysis_df.std() == 0]

if not constant_columns.empty:
    print(f"\nWarning: Dropping constant columns due to zero variance: {list(constant_columns)}")
    analysis_df = analysis_df.drop(columns=constant_columns)

# 5. Calculate correlations
# Calculate the correlation of all columns with the final 'Output' column
#correlations_with_output = analysis_df.drop(columns=['Output']).corrwith(analysis_df['Output'])

# 5. Calculate correlations and sort them
# Make sure the 'Output' column name matches your actual DB column name
OUTPUT_COLUMN_NAME = 'Relative_Polar_Coordinates_1' # <-- CHANGE IF NECESSARY
OUTPUT_COLUMN_NAMES = ['Relative_Polar_Coordinates_0', 'Relative_Polar_Coordinates_1'] 

# Calculate correlations
correlations_with_output = analysis_df.drop(columns=OUTPUT_COLUMN_NAMES).corrwith(
    analysis_df[OUTPUT_COLUMN_NAME],
    method='pearson' # Use 'spearman' or 'pearson' here
)

# Save the results to a CSV file
output_filename = 'correlation_results.csv'
correlations_with_output.sort_values(ascending=False).to_csv(output_filename, header=['InputID,Correlation'])

print(f"\nSaved all correlations to {output_filename}")

# 6. Print and interpret the results commented out
#print("\nCorrelation of each input variable with the Output:")
#print(correlations_with_output.sort_values(ascending=False))

# Sort correlations by absolute value to find the strongest relationships (positive or negative)
top_5_correlated = correlations_with_output.abs().sort_values(ascending=False).head(5)
top_5_feature_names = top_5_correlated.index.tolist()

print(f"\nTop 5 most correlated inputs are: {top_5_feature_names}")

# 6. Generate Scatter Plots using Seaborn
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

for i, feature in enumerate(top_5_feature_names):
    plt.subplot(2, 3, i + 1) # Create a 2x3 grid of plots
    # Use seaborn regplot to draw a scatter plot with a linear regression line
    sns.regplot(x=feature, y=OUTPUT_COLUMN_NAME, data=analysis_df)
    correlation_value = correlations_with_output[feature]
    plt.title(f"{feature} vs Output\nCorr: {correlation_value:.3f}")
    plt.tight_layout()

plt.show()
