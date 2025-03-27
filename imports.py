import subprocess
import sys

# List of required modules
required_modules = [
    "pandas",
    "numpy",
    "seaborn",
    "tensorflow",
    "scikit-learn",
    "streamlit"
]

def install_modules():
    for module in required_modules:
        try:
            __import__(module)  # Try importing the module
        except ImportError:
            print(f"Installing {module}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", module])

if __name__ == "__main__":
    install_modules()
    print("All required modules are installed.")
