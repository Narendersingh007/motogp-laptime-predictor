#!/usr/bin/env python3
"""
Setup script for MotoGP Lap Time Predictor
Installs required dependencies and sets up the environment
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    print("🏍️ Setting up MotoGP Lap Time Predictor...")
    print("=" * 50)
    
    # Check if requirements.txt exists
    if os.path.exists("requirements.txt"):
        print("📦 Installing from requirements.txt...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ All dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install some dependencies. Trying individual installation...")
            
            # Fallback: install packages individually
            packages = [
                "streamlit>=1.28.0",
                "pandas>=1.5.3", 
                "numpy>=1.24.3",
                "scikit-learn>=1.3.0",
                "xgboost>=1.7.4",
                "joblib>=1.2.0",
                "plotly>=5.15.0",
                "seaborn>=0.12.2",
                "matplotlib>=3.7.1",
                "optuna>=3.2.0"
            ]
            
            failed_packages = []
            for package in packages:
                if not install_package(package):
                    failed_packages.append(package)
            
            if failed_packages:
                print(f"\n⚠️ Failed to install: {', '.join(failed_packages)}")
                print("Please install these manually using: pip install <package_name>")
            else:
                print("✅ All packages installed successfully!")
    else:
        print("❌ requirements.txt not found!")
        return False
    
    # Create .streamlit directory if it doesn't exist
    if not os.path.exists(".streamlit"):
        os.makedirs(".streamlit")
        print("📁 Created .streamlit directory")
    
    print("\n🚀 Setup complete! You can now run:")
    print("   streamlit run app.py")
    print("\n🌐 The app will be available at: http://localhost:8501")
    
    return True

if __name__ == "__main__":
    main()