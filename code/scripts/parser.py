import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path, start=None, end=None, weight = 1.7):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""

        start_page = 0
        end_page = reader.get_num_pages()
        if start:
            start_page = start
        if end:
            end_page = end
        
        for page_num in range(start_page, end_page+1):
            page = reader.pages[page_num]
            text += page.extract_text(extraction_mode="layout",
                                      layout_mode_scale_weight=weight)
    return text

def list_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        files = [os.path.join(directory_path, f) for f in files if os.path.isfile(os.path.join(directory_path, f))]
        return files
    except Exception as e:
        print(f"An error occured: {e}")
    return []
