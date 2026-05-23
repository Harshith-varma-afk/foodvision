#!/usr/bin/env python3
"""
Quick launcher script for Food Vision Interface
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit interface."""
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit is not installed!")
        print("Please install it using: pip install streamlit")
        print("Or install all requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check if model file exists
    model_file = "best (1).pt"
    if not os.path.exists(model_file):
        print(f"⚠️  Warning: Model file '{model_file}' not found.")
        print("   The interface will use the default YOLOv8n model.")
        print("   Make sure your trained model is in the current directory.")
        print()
    
    # Launch Streamlit
    print("🚀 Starting Food Vision Interface...")
    print("📱 The interface will open in your default web browser.")
    print("🛑 Press Ctrl+C to stop the server.\n")
    
    try:
        subprocess.run(["streamlit", "run", "foodvision_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Interface stopped. Goodbye!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

