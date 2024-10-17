import os
import re

from pypdf import PdfReader

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

def extract_text_from_pdf(pdf_path, start_page, end_page):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""

        for page_num in range(start_page, end_page+1):
            page = reader.pages[page_num]
            text += page.extract_text(extraction_mode="layout",
                                      layout_mode_scale_weight=1.7)
    return text

def list_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        files = [os.path.join(directory_path, f) for f in files if os.path.isfile(os.path.join(directory_path, f))]
        return files
    except Exception as e:
        print(f"An error occured: {e}")
    return []
