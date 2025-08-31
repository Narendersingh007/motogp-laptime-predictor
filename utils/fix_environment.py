#!/usr/bin/env python3
"""
Environment Fix Script for MotoGP Predictor
Diagnoses and fixes common setup issues
"""

import subprocess
import sys
import os

def run_command(command, description=""):
    """Run a command and return success status"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def check_package(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        print(f"✅ {package_name} - Already installed")
        return True
    except ImportError:
        print(f"❌ {package_name} - Not installed")
        return False

def main():
    print("🏍️ MotoGP Predictor Environment Fix")
    print("=" * 50)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Check if we're in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Not in virtual environment (recommended but not required)")
    
    print("\n📦 Checking required packages...")
    
    required_packages = [
        'pandas', 'numpy', 'streamlit', 'joblib', 
        'sklearn', 'xgboost', 'plotly', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        package_name = 'sklearn' if package == 'scikit-learn' else package
        if not check_package(package_name):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")
        
        # Try installing missing packages
        install_command = f"pip install {' '.join(missing_packages)}"
        if run_command(install_command, f"Installing {', '.join(missing_packages)}"):
            print("✅ All packages installed successfully!")
        else:
            print("❌ Some packages failed to install. Try manual installation:")
            for pkg in missing_packages:
                print(f"   pip install {pkg}")
    else:
        print("✅ All required packages are installed!")
    
    # Fix config file if it exists
    config_path = ".streamlit/config.toml"
    if os.path.exists(config_path):
        print(f"\n🔧 Checking {config_path}...")
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Fix the $PORT issue
            if '$PORT' in content:
                fixed_content = content.replace('port = $PORT\n', '')
                with open(config_path, 'w') as f:
                    f.write(fixed_content)
                print("✅ Fixed config.toml port issue")
            else:
                print("✅ Config file is okay")
                
        except Exception as e:
            print(f"❌ Error checking config: {e}")
    
    print("\n🚀 Environment check complete!")
    print("\nNow try running:")
    print("   streamlit run app_fixed.py")
    print("\nOr if that doesn't work:")
    print("   python -m streamlit run app_fixed.py")

if __name__ == "__main__":
    main()