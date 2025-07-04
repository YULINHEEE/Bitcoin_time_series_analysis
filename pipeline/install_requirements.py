#!/usr/bin/env python3
"""
Installation script for Bitcoin Time Series Analysis project
This script will install all required Python packages
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    print("Bitcoin Time Series Analysis - Package Installation")
    print("=" * 60)
    
    # List of required packages
    required_packages = [
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
        "scikit-learn>=1.0.0",
        "openpyxl>=3.0.0"
    ]
    
    print("Installing required packages...")
    print("-" * 40)
    
    failed_packages = []
    
    for package in required_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    print("\n" + "=" * 60)
    print("Installation Summary")
    print("=" * 60)
    
    if failed_packages:
        print(f"❌ Failed to install {len(failed_packages)} packages:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\nPlease try installing them manually using:")
        print("pip install " + " ".join(failed_packages))
    else:
        print("✅ All packages installed successfully!")
        print("\nYou can now run the analysis with:")
        print("python assignment2.py")
        print("or")
        print("python run_analysis.py")
    
    print("\nNote: If you're using a virtual environment, make sure it's activated.")

if __name__ == "__main__":
    main()
