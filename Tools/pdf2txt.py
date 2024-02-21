import os
import fitz  # PyMuPDF

def convert_pdf_folder_to_txt(source_folder_path, destination_folder_path):
    # Ceck destination folder exists, if not, create it
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)
    
    # Iterate through all files in the source folder
    for filename in os.listdir(source_folder_path):
        if filename.endswith(".pdf"):
            # Construct full file path for the PDF
            pdf_path = os.path.join(source_folder_path, filename)
            # Open the PDF file
            with fitz.open(pdf_path) as doc:
                text = ""
                # Extract text from each page
                for page in doc:
                    text += page.get_text()
                
                # Construct text file name and path in the destination folder
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                txt_path = os.path.join(destination_folder_path, txt_filename)
                
                # Save the extracted text to a .txt file
                with open(txt_path, "w", encoding="utf-8") as text_file:
                    text_file.write(text)
            print(f"Converted {filename} to {txt_filename} in {destination_folder_path}")

# Specify folder paths here
source_folder_path = 'Data Mining files/'
destination_folder_path = 'txt_files/'
convert_pdf_folder_to_txt(source_folder_path, destination_folder_path)
