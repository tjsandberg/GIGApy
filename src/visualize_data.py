import mysql.connector
import pandas as pd
import plotly.express as px
import os
import getpass # Import the getpass module
from dotenv import load_dotenv

# Load environment variables for user, host, and database name
load_dotenv()

# Retrieve credentials from environment variables, but prompt for password
DB_CONFIG = {
    'user': 'root',
    'host': '127.0.0.1', # e.g., 'localhost'
    'database': 'tGigaLocal',
#    'user': 'gigadev',
#   'host': 'giga-db-1-instance-1.c9kea8w2c1dc.us-east-2.rds.amazonaws.com', # e.g., 'localhost'
#   'database': 'tom_2',
#    'user': os.getenv('DB_USER'),
#    'host': os.getenv('DB_HOST'),
#    'database': os.getenv('DB_NAME'),
    'password': None # We will fill this in at runtime
}
TABLE_NAME = 'filteredByWeighted'
TIMESTAMP_COLUMN = 'TimeBorn'
DATA_COLUMNS = ['WeightedFitness', 'RawFitness', 'ValidationFitness', 'OverallFitness']

def fetch_data_from_mysql():
    """Connects to MySQL, fetches data, and returns a pandas DataFrame."""
    
    # Prompt the user for the password securely before attempting connection
    if DB_CONFIG['password'] is None:
        try:
            # This prompts the user and hides the input
            DB_CONFIG['password'] = getpass.getpass(f"Enter password for MySQL user '{DB_CONFIG['user']}' at {DB_CONFIG['host']}: ")
        except Exception as e:
            print(f"Error getting password: {e}")
            return pd.DataFrame()

    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        # query = f"SELECT `{TIMESTAMP_COLUMN}`, `{', '.join(DATA_COLUMNS)}` FROM `{TABLE_NAME}` ORDER BY `{TIMESTAMP_COLUMN}`"
        columns_list_str = ', '.join(f'`{col}`' for col in DATA_COLUMNS)
        query = f"SELECT `{TIMESTAMP_COLUMN}`, {columns_list_str} FROM `{TABLE_NAME}` ORDER BY `{TIMESTAMP_COLUMN}`"
        
        # print(f"Executing Query: {query}") # Optional: uncomment to see the exact query being run

        df = pd.read_sql(query, conn)
        
        df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
        return df

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return pd.DataFrame()
    finally:
        if conn and conn.is_connected():
            conn.close()
            print("MySQL connection closed.")

def generate_plotly_graph(df):
    """Generates and displays an interactive Plotly graph."""
    if df.empty:
        print("Cannot generate graph, DataFrame is empty.")
        return

    df_long = df.melt(
        id_vars=TIMESTAMP_COLUMN, 
        value_vars=DATA_COLUMNS,
        var_name='Metric', 
        value_name='Fitness Value'
    )
    
    fig = px.line(
        df_long, 
        x=TIMESTAMP_COLUMN, 
        y='Fitness Value', 
        color='Metric',
        title=f'Fitness Metrics Over Time from {TABLE_NAME}',
        labels={TIMESTAMP_COLUMN: 'Time Born', 'Fitness Value': 'Fitness Value'}
    )
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Fitness Value",
        legend_title="Fitness Type",
        xaxis_tickformat="%Y-%m-%d %H:%M:%S"
    )
    
#    Uncomment below to Save to file
#    fig.write_html("/home/tom/Personal/fun/Giga/fitness_metrics_graph.html")

    fig.show()

if __name__ == "__main__":
    data_df = fetch_data_from_mysql()
    if not data_df.empty:
        generate_plotly_graph(data_df)


