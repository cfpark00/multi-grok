import subprocess
from concurrent.futures import ThreadPoolExecutor
import sys
import glob

def run_yaml(yaml_path):
    """Function to execute a script using subprocess."""
    try:
        # Ensure your script has the appropriate executable permissions
        result = subprocess.run(['python3', 'run.py', yaml_path], check=True, text=True, capture_output=True)
        print(f"{yaml_path} ran successfully:")
    except subprocess.CalledProcessError as e:
        error_message=e.stderr.decode()
        print(f"Error running {yaml_path}: {error_message}:")

if __name__ == "__main__":
    fol= sys.argv[1]
    yaml_paths = glob.glob(f"{fol}/*.yaml")
    if len(sys.argv) > 2:
        n_par=int(sys.argv[2])
    else:
        n_par=len(yaml_paths)
    print("Running:",yaml_paths)

    # Use ThreadPoolExecutor to run scripts in parallel
    with ThreadPoolExecutor(max_workers=n_par) as executor:
        # Map each script to the executor
        results = executor.map(run_yaml, yaml_paths)
    print("Done")