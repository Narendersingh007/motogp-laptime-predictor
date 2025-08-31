#!/usr/bin/env python3
"""
Debug script to check import issues
"""

import sys
import os

print("🔍 Debugging Python Import Issues")
print("=" * 50)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path:")
for path in sys.path:
    print(f"  - {path}")

print("\n📦 Testing Package Imports:")

packages_to_test = [
    ('joblib', 'joblib'),
    ('plotly', 'plotly.express'),
    ('sklearn', 'sklearn.preprocessing'),
    ('matplotlib', 'matplotlib.pyplot'),
    ('seaborn', 'seaborn'),
    ('xgboost', 'xgboost'),
    ('optuna', 'optuna')
]

for name, import_path in packages_to_test:
    try:
        exec(f"import {import_path}")
        print(f"✅ {name} - OK")
    except ImportError as e:
        print(f"❌ {name} - FAILED: {e}")
    except Exception as e:
        print(f"⚠️  {name} - ERROR: {e}")

print("\n🧪 Testing Virtual Environment:")
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("✅ Running in virtual environment")
    print(f"   Virtual env prefix: {sys.prefix}")
    print(f"   Base prefix: {getattr(sys, 'base_prefix', 'Not available')}")
else:
    print("⚠️  Not running in virtual environment")

print("\n🔧 Site-packages location:")
import site
print(f"Site-packages: {site.getsitepackages()}")