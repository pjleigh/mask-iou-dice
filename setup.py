# installs all required packages

import subprocess
import sys

# Name of the requirements file
requirements_file = 'requirements.txt'

def install_packages(requirements_file):
    try:
        # Use pip to install the packages from the requirements file
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
        print("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    install_packages(requirements_file)
