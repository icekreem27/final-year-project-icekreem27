import json

# Set file paths for two input files and one output file
QA_file_path1 = 'Datasets/QA_Pairs/pdf_QA_Pairs.json'
QA_file_path2 = 'Datasets/QA_Pairs/pptx_QA_Pairs.json'
output_file_path = 'Datasets/QA_Pairs/QA_dataset.jsonl'

# Function to read and load data from a file
def read_data(file_path):
    with open(file_path, 'r') as input_file:
        data = json.load(input_file)
    return data

# Read data from both files
data1 = read_data(QA_file_path1)
data2 = read_data(QA_file_path2)

# Combine data from both files
combined_data = data1 + data2

# Open the output file. Use 'w' to overwrite existing content
with open(output_file_path, 'w') as output_file:
    # Iterate over each item in the combined data
    for item in combined_data:
        # Prepare the formatted structure
        formatted_data = {
            "messages": [
                {"role": "system", "content": "You are a factual chatbot that will help univserity students learn about the COMP2121 Data Mining module"},
                {"role": "user", "content": item['question']},
                {"role": "assistant", "content": item['answer']}
            ]
        }
        
        # Write the formatted data to the output file
        output_file.write(json.dumps(formatted_data) + '\n')
