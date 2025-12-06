import pandas as pd
import numpy as np
import os
import kagglehub
import sqlite3

def download_basketball_dataset():
    """
    Download basketball dataset from Kaggle using kagglehub
    
    Returns:
    --------
    path : str
        Path to downloaded dataset files
    """
    print("=== DOWNLOADING BASKETBALL DATASET ===")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("wyattowalsh/basketball")
        print(f"✓ Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        raise

def list_available_files(dataset_path):
    """
    List all files in the downloaded dataset
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset directory
        
    Returns:
    --------
    files : list
        List of file information dictionaries
    """
    print(f"\n=== FILES IN DATASET ===")
    
    files = []
    for root, dirs, filenames in os.walk(dataset_path):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            files.append({
                'filename': filename,
                'path': filepath,
                'size_mb': round(file_size, 2)
            })
            print(f"  - {filename} ({file_size:.2f} MB)")
    
    return files

def find_database(dataset_path):
    """
    Find SQLite database file in dataset
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset directory
        
    Returns:
    --------
    db_file : str or None
        Path to database file
    """
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.sqlite') or file.endswith('.db'):
                return os.path.join(root, file)
    return None

def get_table_names(db_file):
    """
    Get list of all tables in SQLite database
    
    Parameters:
    -----------
    db_file : str
        Path to database file
        
    Returns:
    --------
    tables : list
        List of table names
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables

def load_from_sqlite(dataset_path, table_name=None):
    """
    Load data from SQLite database
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset directory
    table_name : str or None
        Specific table to load, or None to list all tables
        
    Returns:
    --------
    df : DataFrame or list
        Loaded data or list of table names
    """
    print(f"\n=== LOADING FROM SQLITE DATABASE ===")
    
    # Find database file
    db_file = find_database(dataset_path)
    
    if not db_file:
        print("✗ No SQLite database found in dataset!")
        return None
    
    print(f"✓ Found database: {db_file}")
    
    # Get table names
    table_names = get_table_names(db_file)
    
    # If no specific table requested, return list of tables
    if table_name is None:
        conn = sqlite3.connect(db_file)
        print(f"\nAvailable tables:")
        for table in table_names:
            # Get row count for each table
            count_df = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn)
            print(f"  - {table}: {count_df['count'][0]} rows")
        conn.close()
        return table_names
    
    # Load specific table
    if table_name not in table_names:
        print(f"✗ Table '{table_name}' not found!")
        print(f"Available tables: {table_names}")
        return None
    
    print(f"Loading table: {table_name}")
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def load_with_sql_query(dataset_path, sql_query):
    """
    Execute custom SQL query on the basketball database
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset directory
    sql_query : str
        SQL query to execute
        
    Returns:
    --------
    df : DataFrame
        Query results
    """
    print(f"\n=== EXECUTING SQL QUERY ===")
    print(f"Query: {sql_query[:100]}...")  # Show first 100 chars
    
    # Find database
    db_file = find_database(dataset_path)
    
    if not db_file:
        raise FileNotFoundError("No SQLite database found!")
    
    # Execute query
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    
    print(f"✓ Query returned {len(df)} rows, {len(df.columns)} columns")
    return df

def load_player_stats_for_ml(dataset_path, min_games=20, season=None):
    """
    Load and prepare player statistics specifically for ML
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset directory
    min_games : int
        Minimum games played filter
    season : int or None
        Specific season (e.g., 2023) or None for all seasons
        
    Returns:
    --------
    df : DataFrame
        ML-ready player statistics
    """
    print(f"\n=== LOADING PLAYER STATS FOR ML ===")
    print(f"Filters: min_games={min_games}, season={season}")
    
    # Find database
    db_file = find_database(dataset_path)
    
    if not db_file:
        raise FileNotFoundError("No SQLite database found!")
    
    conn = sqlite3.connect(db_file)
    
    # Check which table exists
    table_names = get_table_names(db_file)
    
    if 'player_season' in table_names:
        table = 'player_season'
        games_col = 'games'
    elif 'Player' in table_names:
        table = 'Player'
        games_col = 'games'  # Adjust if different
    else:
        print(f"Available tables: {table_names}")
        raise ValueError("Could not find player statistics table!")
    
    # Build query
    query = f"SELECT * FROM {table} WHERE {games_col} >= {min_games}"
    
    if season:
        query += f" AND season = {season}"
    
    print(f"SQL: {query}")
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"✓ Loaded {len(df)} player-seasons")
    return df

def load_basketball_data_complete(use_cached=True):
    """
    Complete data loading pipeline:
    1. Download from Kaggle
    2. Find and connect to database
    3. Load player statistics for ML
    
    Parameters:
    -----------
    use_cached : bool
        If True, use previously downloaded data if available
        
    Returns:
    --------
    tuple : (DataFrame, str)
        - DataFrame: Complete basketball dataset ready for ML
        - str: Path to the downloaded dataset
    """
    print("=== COMPLETE BASKETBALL DATA LOADING ===\n")
    
    # Step 1: Download dataset
    dataset_path = download_basketball_dataset()
    
    # Step 2: List available files
    files = list_available_files(dataset_path)
    
    # Step 3: Find database
    db_file = find_database(dataset_path)
    if not db_file:
        raise FileNotFoundError("No SQLite database found in dataset!")
    
    print(f"\n✓ Found database: {db_file}")
    
    # Step 4: Get available tables
    table_names = get_table_names(db_file)
    print(f"\n=== AVAILABLE TABLES ===")
    
    conn = sqlite3.connect(db_file)
    for table in table_names:
        count_df = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn)
        print(f"  - {table}: {count_df['count'][0]} rows")
    
    # Step 5: Load best table for ML
    df = None
    preferred_tables = ['player_season', 'Player', 'player', 'players']
    
    for table_name in preferred_tables:
        if table_name in table_names:
            print(f"\n=== LOADING TABLE: {table_name} ===")
            
            # Test query first
            try:
                test_query = f"SELECT * FROM {table_name} LIMIT 5"
                test_df = pd.read_sql_query(test_query, conn)
                print(f"✓ Test successful: {len(test_df.columns)} columns")
                
                # Load with filter if it's player_season
                if table_name == 'player_season':
                    # Check if 'games' column exists
                    if 'games' in test_df.columns:
                        query = f"SELECT * FROM {table_name} WHERE games >= 20"
                        print(f"  Applying filter: games >= 20")
                    else:
                        query = f"SELECT * FROM {table_name}"
                else:
                    query = f"SELECT * FROM {table_name}"
                
                df = pd.read_sql_query(query, conn)
                print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
                break
                
            except Exception as e:
                print(f"✗ Error loading {table_name}: {e}")
                continue
    
    conn.close()
    
    # Fallback to first table if nothing worked
    if df is None and len(table_names) > 0:
        print(f"\n⚠ Loading first available table: {table_names[0]}")
        conn = sqlite3.connect(db_file)
        df = pd.read_sql_query(f"SELECT * FROM {table_names[0]}", conn)
        conn.close()
    
    if df is None:
        raise ValueError("Could not load any data from database!")
    
    print(f"\n{'='*50}")
    print(f"✓ DATASET LOADED SUCCESSFULLY!")
    print(f"{'='*50}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)[:10]}...")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df, dataset_path

def explore_table_schema(dataset_path, table_name):
    """
    Show column information for a specific table
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset
    table_name : str
        Table to explore
    """
    print(f"\n=== SCHEMA FOR TABLE: {table_name} ===")
    
    db_file = find_database(dataset_path)
    if not db_file:
        print("✗ No database found!")
        return
    
    conn = sqlite3.connect(db_file)
    
    # Get table info
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    schema = cursor.fetchall()
    
    print(f"\nColumns in {table_name}:")
    for col in schema:
        pk_marker = " [PRIMARY KEY]" if col[5] else ""
        print(f"  {col[1]:30} {col[2]:15}{pk_marker}")
    
    # Sample data
    print(f"\nSample rows:")
    sample = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
    print(sample)
    
    conn.close()

def get_dataset_info(df):
    """
    Print comprehensive information about the dataset
    
    Parameters:
    -----------
    df : DataFrame
        Basketball dataset
    """
    print("\n=== DATASET INFORMATION ===")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nBasic statistics:\n{df.describe()}")

# Example usage (commented out - uncomment to test)
if __name__ == "__main__":
    print("Testing data loader...")
    try:
        df, path = load_basketball_data_complete()
        print("\n✓ Test successful!")
        print(f"First few rows:\n{df.head()}")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()