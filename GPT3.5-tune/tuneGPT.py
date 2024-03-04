# api_key = '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'
from openai import OpenAI

def create_file(file_path):
    response = client.files.create(
        file = open(file_path, "rb"),
        purpose = "fine-tune"
    )
    
    file_id = response.id
    return file_id

def create_fine_tune(file_name):
    client.fine_tuning.jobs.create(
        training_file = file_name, 
        model = "gpt-3.5-turbo",
        hyperparameters = {
            "n_epochs" : 10
        }
    )


# main
client = OpenAI()
file_path = "Datasets/final_QA.jsonl"

# create file on OpenAI
fileID = create_file(file_path)
print("File ID:", fileID)

# create tune job
create_fine_tune(fileID)
