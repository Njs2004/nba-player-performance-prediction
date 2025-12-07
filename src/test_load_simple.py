from src.data_loader import load_basketball_data_complete, explore_table_schema

print("="*70)
print("TESTING NBA DATA LOADER")
print("="*70)

# Load data
df, path = load_basketball_data_complete(min_games=30)

print("\n✓ Data loaded successfully!")
print(f"\nDataset columns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\nSample data:")
print(df.head())

# Explore the table schema
explore_table_schema(path, 'player_season')