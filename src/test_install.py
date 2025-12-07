# test_install.py
print("Testing package installations...\n")

packages = [
    'numpy',
    'pandas', 
    'matplotlib',
    'seaborn',
    'sklearn',
    'kagglehub'
]

for package in packages:
    try:
        __import__(package)
        print(f"✓ {package} installed")
    except ImportError:
        print(f"✗ {package} NOT installed - run: pip3 install {package}")

print("\nTest complete!")