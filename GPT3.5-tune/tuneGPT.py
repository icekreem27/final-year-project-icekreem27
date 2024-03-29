# api_key = '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'

from openai import OpenAI

def create_file(file_path):
    response = client.files.create(
        file=open(file_path, "rb"),
        purpose="fine-tune"
    )
    
    file_id = response.id
    return file_id

def create_fine_tune(training_file_name, validation_file_name):
    client.fine_tuning.jobs.create(
        training_file=training_file_name, 
        validation_file=validation_file_name,
        model="gpt-3.5-turbo",
        suffix="FullDataset-0.25LR",
        hyperparameters={
            "learning_rate_multiplier":0.25
        }
    )

# Initialize the OpenAI client
client = OpenAI()

# File paths for the training and validation datasets
training_file_path = "Datasets/Augmented/full_dataset.jsonl"
validation_file_path = "Datasets/Augmented/validation_dataset.jsonl" 

# Upload the training file to OpenAI
training_file_id = create_file(training_file_path)
print("Training File ID:", training_file_id)

# Upload the validation file to OpenAI
validation_file_id = create_file(validation_file_path)
print("Validation File ID:", validation_file_id)

# Create the fine-tuning job with both training and validation files
create_fine_tune(training_file_id, validation_file_id)
