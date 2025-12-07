import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def handle_missing_values(df, strategy='drop'):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    df : DataFrame
        Dataset with potential missing values
    strategy : str
        'drop' - remove rows with missing values
        'mean' - fill with column mean
        'median' - fill with column median
        
    Returns:
    --------
    df_clean : DataFrame
        Dataset with missing values handled
        
    Why handle missing values?
    - ML algorithms can't process NaN values
    - Missing data can indicate data quality issues
    - Different strategies for different situations:
      * Drop: if few missing values (<5%)
      * Mean: if data is normally distributed
      * Median: if data has outliers
    """
    print(f"\n=== HANDLING MISSING VALUES ===")
    print(f"Missing values before:\n{df.isnull().sum()}")
    
    if strategy == 'drop':
        df_clean = df.dropna()
        print(f"Rows removed: {len(df) - len(df_clean)}")
    elif strategy == 'mean':
        df_clean = df.fillna(df.mean())
        print("Filled with column means")
    elif strategy == 'median':
        df_clean = df.fillna(df.median())
        print("Filled with column medians")
    
    print(f"Missing values after:\n{df_clean.isnull().sum()}")
    return df_clean

def remove_outliers(df, columns, z_threshold=3):
    """
    Remove outliers using Z-score method
    
    Parameters:
    -----------
    df : DataFrame
        Dataset
    columns : list
        Columns to check for outliers
    z_threshold : float
        Z-score threshold (typically 3 = 99.7% of data)
        
    Returns:
    --------
    df_clean : DataFrame
        Dataset with outliers removed
        
    Why remove outliers?
    - Extreme values can skew model predictions
    - May represent data entry errors
    - Example: Player listed with 200 points per game (impossible)
    - Z-score method: values >3 standard deviations from mean
    """
    print(f"\n=== REMOVING OUTLIERS ===")
    print(f"Original size: {len(df)}")
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df.columns:
            # Calculate Z-score
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            # Keep only rows within threshold
            df_clean = df_clean[z_scores < z_threshold]
    
    print(f"After outlier removal: {len(df_clean)}")
    print(f"Rows removed: {len(df) - len(df_clean)}")
    
    return df_clean.reset_index(drop=True)

def create_derived_features(df):
    """
    Create new features from existing ones (Feature Engineering)
    
    Parameters:
    -----------
    df : DataFrame
        Dataset with raw features
        
    Returns:
    --------
    df_enhanced : DataFrame
        Dataset with additional engineered features
        
    Why feature engineering?
    - Create more meaningful predictors
    - Capture domain knowledge (basketball expertise)
    - Examples:
      * Points per minute = efficiency metric
      * True shooting % = better shooting metric
      * Usage rate = how often player is involved
    """
    print(f"\n=== FEATURE ENGINEERING ===")
    df_enhanced = df.copy()
    
    # Points per minute (efficiency)
    if 'points' in df.columns and 'minutes_played' in df.columns:
        df_enhanced['points_per_minute'] = df['points'] / df['minutes_played']
        print("Created: points_per_minute")
    
    # Shooting efficiency
    if 'field_goals_made' in df.columns and 'field_goals_attempted' in df.columns:
        df_enhanced['field_goal_pct'] = df['field_goals_made'] / df['field_goals_attempted']
        print("Created: field_goal_pct")
    
    # Rebound rate (if height available)
    if 'rebounds' in df.columns and 'games' in df.columns:
        df_enhanced['rebounds_per_game'] = df['rebounds'] / df['games']
        print("Created: rebounds_per_game")
    
    # Assist-to-turnover ratio (playmaking skill)
    if 'assists' in df.columns and 'turnovers' in df.columns:
        df_enhanced['assist_to_turnover'] = df['assists'] / (df['turnovers'] + 1)  # +1 to avoid division by zero
        print("Created: assist_to_turnover")
    
    print(f"Total features now: {len(df_enhanced.columns)}")
    return df_enhanced

def encode_categorical_features(df, categorical_cols):
    """
    Convert categorical variables to numerical (One-Hot Encoding)
    
    Parameters:
    -----------
    df : DataFrame
        Dataset with categorical columns
    categorical_cols : list
        List of categorical column names
        
    Returns:
    --------
    df_encoded : DataFrame
        Dataset with categorical variables encoded
        
    Why encode categorical variables?
    - ML algorithms need numerical input
    - One-hot encoding: creates binary columns for each category
    - Example: Position [Guard, Forward, Center] becomes:
      * is_Guard: [1, 0, 0]
      * is_Forward: [0, 1, 0]
      * is_Center: [0, 0, 1]
    """
    print(f"\n=== ENCODING CATEGORICAL FEATURES ===")
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in df.columns:
            # One-hot encode
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(col, axis=1)
            print(f"Encoded: {col} -> {len(dummies.columns)} new columns")
    
    return df_encoded

def split_data(df, target_col, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Parameters:
    -----------
    df : DataFrame
        Full dataset
    target_col : str
        Name of target variable (what we're predicting)
    test_size : float
        Proportion for testing (0.2 = 20%)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Training and testing data
        
    Why split data?
    - Training set: used to train the model
    - Testing set: used to evaluate performance on unseen data
    - 80/20 split is standard practice
    - Random state ensures same split every time (reproducibility)
    """
    print(f"\n=== SPLITTING DATA ===")
    
    # Separate features (X) and target (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Features: {len(X_train.columns)}")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Standardize features (mean=0, std=1)
    
    Parameters:
    -----------
    X_train, X_test : DataFrames
        Training and testing features
        
    Returns:
    --------
    X_train_scaled, X_test_scaled : arrays
        Scaled features
    scaler : StandardScaler object
        Fitted scaler (for inverse transform if needed)
        
    Why scale features?
    - Features on different scales (minutes: 0-40, points: 0-30, etc.)
    - Linear regression sensitive to scale
    - Standardization: (x - mean) / std
    - IMPORTANT: Fit on training data, transform both train and test
      (prevents data leakage - test info shouldn't influence training)
    """
    print(f"\n=== SCALING FEATURES ===")
    
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    print("Fitted scaler on training data")
    
    # Transform test data using training statistics
    X_test_scaled = scaler.transform(X_test)
    print("Transformed test data")
    
    print(f"Mean after scaling: {X_train_scaled.mean():.6f}")
    print(f"Std after scaling: {X_train_scaled.std():.6f}")
    
    return X_train_scaled, X_test_scaled, scaler