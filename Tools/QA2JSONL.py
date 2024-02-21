import json

# Set file paths
QA_file_path = 'Datasets/QA_Pairs.json'
output_file_path = 'Datasets/QA_dataset.jsonl'

# Read the file
with open(QA_file_path, 'r') as input_file:
    data = json.load(input_file)

    # Open the output file. Use 'w' to overwrite existing content or 'a' to append
    with open(output_file_path, 'w') as output_file:
        
        # Iterate over each item in the data
        for item in data:
            # Prepare the formatted structure
            formatted_data = {
                "messages": [
                    {"role": "system", "content": "You are a factual chatbot that will help student learn about the COMP2121 Data Mining module"},
                    {"role": "user", "content": item['question']},
                    {"role": "assistant", "content": item['answer']}
                ]
            }
            
            # Write the formatted data to the output file
            # Ensure it's written as a JSON string with json.dumps and add a newline for each entry if desired
            output_file.write(json.dumps(formatted_data) + '\n')