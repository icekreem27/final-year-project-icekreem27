import json
import nltk
import random
from nltk.corpus import wordnet
from googletrans import Translator
from tqdm import tqdm

# Ensure necessary NLTK data is downloaded for processing
nltk.download('wordnet')

def get_synonyms(word):
    """Get synonyms for a given word but excluding the word itself."""
    # Generates a set of synonyms
    # Replacing underscores and dashes with spaces
    # Converting to lowercase
    return {lemma.name().replace('_', ' ').replace('-', ' ').lower()
            for syn in wordnet.synsets(word)
            for lemma in syn.lemmas()
            if lemma.name().replace('_', ' ').replace('-', ' ').lower() != word}

def synonym_replacement(sentence, n=3):
    """Replace up to n words in the sentence with their synonyms."""
    # Tokenizes the sentence and selects unique, alphanumeric words
    words = nltk.word_tokenize(sentence)
    unique_words = set([word for word in words if word.isalnum()])
    # Randomly picks words to replace
    random_words = random.sample(list(unique_words), min(n, len(unique_words)))
    new_sentence = sentence
    
    for word in random_words:
        synonyms = get_synonyms(word)
        if synonyms:
            # Create new sentence with one of its synonyms
            new_sentence = new_sentence.replace(word, random.choice(list(synonyms)), 1)
    
    return new_sentence

def back_translation(sentence, src_lang='en', intermediate_lang='fr'):
    """Perform back-translation on the sentence through an intermediate language."""
    translator = Translator()
    # Translate from source to intermediate language
    translated_to_intermediate = translator.translate(sentence, src=src_lang, dest=intermediate_lang).text
    # Translate back to source language
    back_translated = translator.translate(translated_to_intermediate, src=intermediate_lang, dest=src_lang).text
    
    return back_translated

def augment_data(jsonl_file_path, output_file_path):
    """Augment a dataset with synonym replacement and back-translation."""
    augmented_data = []
        
    # First pass to count lines for the progress bar
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f)

    # Second pass to process the lines with a progress bar
    augmented_data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        # Initialize the progress bar with the total number of lines
        progress_bar = tqdm(f, total=total_lines, desc="Augmenting data", unit="line")
        for line in progress_bar:
            data = json.loads(line)
            # Extracts questions and answers
            question = data['messages'][1]['content']
            answer = data['messages'][2]['content']
            
            # Performs synonym replacement and back-translation
            syn_question = synonym_replacement(question)
            syn_answer = synonym_replacement(answer)
            bt_question = back_translation(question)
            bt_answer = back_translation(answer)
            
            # Appends original and augmented data to the dataset
            augmented_data.append(data)  # Original data
            augmented_data.append({"messages": [{"role": "system", "content": "You are a factual chatbot that will help students learn about the COMP2121 Data Mining module"}, {"role": "user", "content": syn_question}, {"role": "assistant", "content": syn_answer}]})
            
            # Checks if back translated sentence is the same as orignal
            if bt_question != question or bt_answer != answer:
                augmented_data.append({"messages": [{"role": "system", "content": "You are a factual chatbot that will help students learn about the COMP2121 Data Mining module"}, {"role": "user", "content": bt_question}, {"role": "assistant", "content": bt_answer}]})
    
    # Writes augmented data to file
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(augmented_data, desc="Writing data", unit="item"):
            f_out.write(json.dumps(item) + '\n')

def split_data(full_dataset_path, training_file_path, validation_file_path, validation_split=0.2): # 80/20 split
    """Split the dataset into training and validation sets."""
    # Loads the dataset and shuffles it
    # Avoids removing certain units alltogether
    with open(full_dataset_path, 'r', encoding='utf-8') as f:
        augmented_data = [json.loads(line) for line in f]
    
    random.shuffle(augmented_data)
    # Calculates the size of the validation set
    validation_size = int(len(augmented_data) * validation_split)
    # Splits the data
    validation_data = augmented_data[:validation_size]
    training_data = augmented_data[validation_size:]
    
    # Writes the training and validation data to separate files
    with open(training_file_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(training_data, desc="Writing training data", unit="item"):
            f_out.write(json.dumps(item) + '\n')
    
    with open(validation_file_path, 'w', encoding='utf-8') as f_val:
        for item in tqdm(validation_data, desc="Writing validation data", unit="item"):
            f_val.write(json.dumps(item) + '\n')

# Defines file paths for input and output
jsonl_file_path = 'Datasets/final_QA.jsonl'
full_dataset_path = 'Datasets/Augmented/full_dataset.jsonl'
training_file_path = 'Datasets/Augmented/training_dataset.jsonl'
validation_file_path = 'Datasets/Augmented/validation_dataset.jsonl'

# Executes data augmentation and splits the dataset
augment_data(jsonl_file_path, full_dataset_path)
split_data(full_dataset_path, training_file_path, validation_file_path)
