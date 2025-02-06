import os
import re
from parser import extract_text_from_pdf, list_files_in_directory

def clean_blood_memory(text):
    # Reduce excess white space to a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove capitalized phrases "MARTHA GRAHAM" and "BLOOD MEMORY"
    text = re.sub(r'\bMARTHA GRAHAM\b', '', text)
    text = re.sub(r'\bBLOOD MEMORY\b', '', text)

    # Define regex pattern to match sequences of digits from 1 to 500, with optional spaces between digits
    pattern = r'\b(?:[1-9]|[1-4][0-9]|[1-4] [0-9]|50|5 0 0|[1-9] [0-9]|[1-9] [0-9] [0-9]|[1-4][0-9][0-9]|[1-4] [0-9][0-9]|[1-4][0-9] [0-9]|[1-4] [0-9] [0-9])\b' 
    
    text = re.sub(pattern, '', text)

    # Strip leading and trailing white space
    text = text.strip()
    
    return text

def write_raw_output_files(input, output, n_workers=1):
    files = list_files_in_directory(input)
    # Loop could be parallelized
    for file in files:
        text = extract_text_from_pdf(file)

        text_filename = os.path.splitext(file)[0] + ".txt"
        text_filepath = os.path.join(output, text_filename)
        print(f"Writing {text_filename} to {text_filepath}")
        with open(text_filepath, "w") as text_handle:
            text_handle.write(text)