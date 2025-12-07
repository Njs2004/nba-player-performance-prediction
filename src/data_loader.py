import pandas as pd
import numpy as np
import os
import kagglehub
import sqlite3
from typing import Optional, Dict, List, Tuple, Any

def download_basketball_dataset() -> str:
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
        print(f"[SUCCESS] Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"[ERROR] Error downloading dataset: {e}")
        raise

def find_database(dataset_path: str) -> Optional[str]:
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
                db_path = os.path.join(root, file)
                print(f"[SUCCESS] Found database: {db_path}")
                return db_path
    return None

def get_table_names(db_file: str) -> List[str]:
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

def get_table_info(db_file: str, table_name: str) -> Dict[str, Any]:
    """
    Get information about a specific table
    
    Parameters:
    -----------
    db_file : str
        Path to database file
    table_name : str
        Name of table
        
    Returns:
    --------
    info : dict
        Dictionary with row_count and columns
    """
    conn = sqlite3.connect(db_file)
    
    # Get row count
    count_df = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table_name}", conn)
    row_count = count_df['count'][0]
    
    # Get column names
    sample = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 1", conn)
    columns = sample.columns.tolist()
    
    conn.close()
    
    return {
        'row_count': row_count,
        'columns': columns
    }

def list_all_tables(dataset_path: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    List all tables in the basketball database with details
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset directory
        
    Returns:
    --------
    tables_info : dict or None
        Dictionary with table information, or None if no database found
    """
    print("\n=== BASKETBALL DATABASE TABLES ===")
    
    db_file = find_database(dataset_path)
    if not db_file:
        print("[ERROR] No database found!")
        return None
    
    table_names = get_table_names(db_file)
    tables_info: Dict[str, Dict[str, Any]] = {}
    
    for table in table_names:
        info = get_table_info(db_file, table)
        tables_info[table] = info
        print(f"  * {table:30} {info['row_count']:>8,} rows  |  {len(info['columns']):>3} columns")
    
    return tables_info

def load_table(dataset_path: str, table_name: str, limit: Optional[int] = None, 
               where_clause: Optional[str] = None) -> pd.DataFrame:
    """
    Load a specific table from the database
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset directory
    table_name : str
        Name of table to load
    limit : int or None
        Maximum rows to load (None = all rows)
    where_clause : str or None
        SQL WHERE clause (e.g., "games >= 20")
        
    Returns:
    --------
    df : DataFrame
        Loaded data
    """
    print(f"\n=== LOADING TABLE: {table_name} ===")
    
    db_file = find_database(dataset_path)
    if not db_file:
        raise FileNotFoundError("No database found!")
    
    # Build query
    query = f"SELECT * FROM {table_name}"
    if where_clause:
        query += f" WHERE {where_clause}"
    if limit:
        query += f" LIMIT {limit}"
    
    print(f"SQL: {query}")
    
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"[SUCCESS] Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def load_with_sql_query(dataset_path: str, sql_query: str) -> pd.DataFrame:
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
    print(f"\n=== EXECUTING CUSTOM SQL QUERY ===")
    
    db_file = find_database(dataset_path)
    if not db_file:
        raise FileNotFoundError("No database found!")
    
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    
    print(f"[SUCCESS] Query returned {len(df)} rows, {len(df.columns)} columns")
    return df

def explore_table_schema(dataset_path: str, table_name: str) -> None:
    """
    Show detailed schema for a specific table
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset
    table_name : str
        Table to explore
    """
    print(f"\n{'='*70}")
    print(f"SCHEMA FOR TABLE: {table_name}")
    print(f"{'='*70}")
    
    db_file = find_database(dataset_path)
    if not db_file:
        print("[ERROR] No database found!")
        return
    
    conn = sqlite3.connect(db_file)
    
    # Get table schema
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    schema = cursor.fetchall()
    
    print("\nColumns:")
    print(f"{'Name':<30} {'Type':<15} {'Nullable':<10} {'Default':<15} {'PK':<5}")
    print("-" * 80)
    for col in schema:
        col_name = col[1]
        col_type = col[2]
        not_null = "NO" if col[3] else "YES"
        default_val = col[4] if col[4] else ""
        is_pk = "YES" if col[5] else ""
        print(f"{col_name:<30} {col_type:<15} {not_null:<10} {str(default_val):<15} {is_pk:<5}")
    
    # Get sample data
    print(f"\n{'='*70}")
    print("SAMPLE DATA (first 5 rows):")
    print(f"{'='*70}")
    sample = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
    print(sample.to_string())
    
    # Get statistics
    print(f"\n{'='*70}")
    print("STATISTICS:")
    print(f"{'='*70}")
    count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table_name}", conn)
    print(f"Total rows: {count['count'][0]:,}")
    
    conn.close()

def load_basketball_data_complete(min_games: int = 20, 
                                  season: Optional[int] = None,
                                  max_rows: int = 50000) -> Tuple[pd.DataFrame, str]:
    """
    Complete data loading pipeline - automatically finds and loads best table
    OPTIMIZED FOR MACBOOK AIR - Skips huge tables like play-by-play
    
    Parameters:
    -----------
    min_games : int
        Minimum games played filter (default: 20)
    season : int or None
        Specific season to load (e.g., 2023), None for all seasons
    max_rows : int
        Skip tables with more than this many rows (default: 50,000)
        
    Returns:
    --------
    tuple : (DataFrame, str)
        - DataFrame: Player statistics ready for ML
        - str: Path to the downloaded dataset
    """
    print("\n" + "="*70)
    print("NBA PLAYER PERFORMANCE PREDICTION - DATA LOADING")
    print("(MacBook Air Optimized - Skipping Large Tables)")
    print("="*70)
    
    # Step 1: Download dataset
    dataset_path = download_basketball_dataset()
    
    # Step 2: Find database
    db_file = find_database(dataset_path)
    if not db_file:
        raise FileNotFoundError("No SQLite database found in dataset!")
    
    # Step 3: List all tables
    tables_info = list_all_tables(dataset_path)
    
    # Check if tables_info is valid
    if tables_info is None:
        raise ValueError("No tables found in database!")
    
    if len(tables_info) == 0:
        raise ValueError("Database contains no tables!")
    
    # Filter out HUGE tables (like play-by-play)
    print(f"\n[OPTIMIZATION] Filtering tables for MacBook Air...")
    print(f"[OPTIMIZATION] Skipping tables with > {max_rows:,} rows")
    
    filtered_tables = {}
    skipped_tables = []
    
    for table_name, info in tables_info.items():
        if info['row_count'] > max_rows:
            skipped_tables.append((table_name, info['row_count']))
            print(f"  [SKIP] {table_name}: {info['row_count']:,} rows (too large)")
        else:
            filtered_tables[table_name] = info
            print(f"  [OK] {table_name}: {info['row_count']:,} rows")
    
    if len(filtered_tables) == 0:
        raise ValueError("All tables are too large! Try increasing max_rows parameter.")
    
    print(f"\n[INFO] Using {len(filtered_tables)} tables, skipped {len(skipped_tables)}")
    
    # Step 4: Try to find the best table for player statistics
    conn = sqlite3.connect(db_file)
    table_names = list(filtered_tables.keys())
    
    # Priority order - most datasets use these names
    preferred_tables = [
        'player_season',      # Most common - perfect for ML
        'player',             # Player info
        'Player',             # Alternative name
        'players',            # Lowercase variant
        'player_stats',       # Alternative
        'Player_Attributes',  # Some datasets
        'team_season',        # Backup: team stats (smaller)
    ]
    
    df: Optional[pd.DataFrame] = None
    loaded_table: Optional[str] = None
    
    print("\n" + "="*70)
    print("FINDING BEST TABLE FOR ML...")
    print("="*70)
    
    for table_name in preferred_tables:
        if table_name in table_names:
            print(f"\n[SUCCESS] Found '{table_name}' table - attempting to load...")
            
            try:
                # Test query to check columns
                test_query = f"SELECT * FROM {table_name} LIMIT 1"
                test_df = pd.read_sql_query(test_query, conn)
                
                print(f"  Columns: {list(test_df.columns)[:10]}...")
                print(f"  Table size: {filtered_tables[table_name]['row_count']:,} rows")
                
                # Build filtered query
                query = f"SELECT * FROM {table_name}"
                filters: List[str] = []
                
                # Check for games column (various names)
                games_cols = ['games', 'gp', 'G', 'games_played']
                games_col: Optional[str] = None
                for col in games_cols:
                    if col in test_df.columns:
                        games_col = col
                        break
                
                if games_col:
                    filters.append(f"{games_col} >= {min_games}")
                    print(f"  [FILTER] Applying filter: {games_col} >= {min_games}")
                
                # Check for season column
                season_cols = ['season', 'year', 'season_id']
                season_col: Optional[str] = None
                for col in season_cols:
                    if col in test_df.columns:
                        season_col = col
                        break
                
                if season and season_col:
                    filters.append(f"{season_col} = {season}")
                    print(f"  [FILTER] Applying filter: {season_col} = {season}")
                
                # Add filters to query
                if filters:
                    query += " WHERE " + " AND ".join(filters)
                
                # Add LIMIT as safety for MacBook Air
                estimated_rows = filtered_tables[table_name]['row_count']
                if estimated_rows > 30000:
                    query += f" LIMIT 30000"
                    print(f"  [SAFETY] Adding LIMIT 30000 for memory safety")
                
                print(f"\n  Executing: {query[:100]}...")
                
                # Load data
                df = pd.read_sql_query(query, conn)
                loaded_table = table_name
                
                print(f"\n  [SUCCESS] Loaded {len(df):,} rows from '{table_name}'")
                print(f"  [MEMORY] Estimated size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                break
                
            except Exception as e:
                print(f"  [ERROR] Error loading '{table_name}': {e}")
                continue
    
    # If no preferred table found, load smallest table
    if df is None:
        print("\n[WARNING] No preferred table found. Loading smallest available table...")
        
        # Find smallest table
        smallest_table = min(filtered_tables.items(), key=lambda x: x[1]['row_count'])
        loaded_table = smallest_table[0]
        
        print(f"[INFO] Attempting to load '{loaded_table}' ({smallest_table[1]['row_count']:,} rows)")
        df = pd.read_sql_query(f"SELECT * FROM {loaded_table} LIMIT 30000", conn)
        print(f"[SUCCESS] Loaded '{loaded_table}' with {len(df):,} rows")
    
    conn.close()
    
    # Final check
    if df is None or len(df) == 0:
        raise ValueError("Failed to load any data from database!")
    
    # Memory check
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    if memory_mb > 500:
        print(f"\n[WARNING] Dataset is large ({memory_mb:.2f} MB)")
        print(f"[WARNING] Consider using a specific season or reducing features")
    
    # Final summary
    print("\n" + "="*70)
    print("[SUCCESS] DATA LOADING COMPLETE!")
    print("="*70)
    print(f"Dataset path: {dataset_path}")
    print(f"Database: {os.path.basename(db_file)}")
    print(f"Table loaded: {loaded_table}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory: {memory_mb:.2f} MB")
    print(f"\nFirst 10 columns: {list(df.columns)[:10]}")
    
    if len(skipped_tables) > 0:
        print(f"\n[INFO] Skipped {len(skipped_tables)} large tables:")
        for table, rows in skipped_tables[:3]:
            print(f"  - {table}: {rows:,} rows")
    
    return df, dataset_path
def get_dataset_info(df: pd.DataFrame) -> None:
    """
    Print comprehensive information about the dataset
    
    Parameters:
    -----------
    df : DataFrame
        Basketball dataset
    """
    print("\n" + "="*70)
    print("DATASET INFORMATION")
    print("="*70)
    
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    print("\n--- Data Types ---")
    print(df.dtypes.value_counts())
    
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_df = missing[missing > 0].sort_values(ascending=False)
        print(missing_df)
    else:
        print("[SUCCESS] No missing values!")
    
    print("\n--- Numeric Columns Summary ---")
    print(df.describe())
    
    print("\n--- Sample Data ---")
    print(df.head(3))

# Test function
if __name__ == "__main__":
    print("Testing basketball data loader...\n")
    try:
        df, path = load_basketball_data_complete(min_games=20)
        print("\n" + "="*70)
        print("[SUCCESS] TEST SUCCESSFUL!")
        print("="*70)
        print(f"\nDataset ready for analysis!")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print("\n" + "="*70)
        print("[ERROR] TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
