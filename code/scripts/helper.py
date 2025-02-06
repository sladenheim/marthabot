import os

def get_tmpdir():
    return os.environ.get("TMPDIR", None)

def add_subdirectory(path, subdirectory):
    # Combine the path and subdirectory to get the full path
    full_path = os.path.join(path, subdirectory)
    
    # Check if the subdirectory exists
    if not os.path.exists(full_path):
        # Create the subdirectory
        os.makedirs(full_path)
        print(f"Subdirectory '{subdirectory}' created at '{path}'")
    else:
        print(f"Subdirectory '{subdirectory}' already exists at '{path}'")

    return full_path