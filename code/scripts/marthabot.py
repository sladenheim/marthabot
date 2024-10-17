import configparser

from parser import extract_text_from_pdf
from parser import list_files_in_directory
from parser import clean_blood_memory

config = configparser.ConfigParser()
config.read('marthabot.cfg')

input_data = config["Data"].get("input_data")
raw_output = config["Data"].get("raw_output")

files = list_files_in_directory(input_data)
for file in files:
    text = extract_text_from_pdf(file, 12, 284)


mg_blood_memory = raw_output + "/blood_memory.txt"
with open(mg_blood_memory, "w") as file:
    file.write(clean_blood_memory(text))