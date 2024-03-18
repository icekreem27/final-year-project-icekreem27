import json
import nltk
import random
import time
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from random import choice, sample
from deep_translator import GoogleTranslator
from tqdm import tqdm

# Ensure necessary NLTK data is downloaded
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(treebank_tag):
    """Converts treebank POS tags to WordNet POS tags."""
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return None

def get_synonyms_pos(word, pos):
    """Get synonyms for a given word and POS but excluding the word itself."""
    synonyms = set()
    for syn in nltk.corpus.wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').replace('-', ' ').lower()
            if synonym != word:
                synonyms.add(synonym)
    return synonyms

def synonym_replacement(sentence, n=3):
    """Replace up to n words in the sentence with their synonyms"""
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)
    
    # Select words that are alphanumeric and have a valid WordNet POS tag
    valid_words_indices = [(index, (word, get_wordnet_pos(pos))) for index, (word, pos) in enumerate(tagged_words) if word.isalnum() and get_wordnet_pos(pos)]
    if not valid_words_indices:
        return sentence  # no valid words to replace
    
    # Randomly pick words to replace
    replace_candidates_indices = sample(valid_words_indices, min(n, len(valid_words_indices)))
    
    replaced_indices = []
    new_words = words[:]
    for index, (word, pos) in replace_candidates_indices:
        if index not in replaced_indices:  # ensure word hasn't been replaced
            synonyms = get_synonyms_pos(word, pos)
            if synonyms:
                chosen_synonym = choice(list(synonyms))
                if chosen_synonym not in new_words:  # Avoid repetition of words
                    new_words[index] = chosen_synonym
                    replaced_indices.append(index)
    
    return ' '.join(new_words)


def back_translation(sentence, src_lang='en', intermediate_langs=['fr', 'de', 'es']):
    """Perform back-translation on the sentence through a randomly selected intermediate language."""
    # Select a random language
    intermediate_lang = random.choice(intermediate_langs)
    
    max_retries = 3  # Maximum number of retries
    retry_delay = 10  # Time to wait between retries
    
    for attempt in range(max_retries):
        try:
            # Translate from source to intermediate language
            translated_to_intermediate = GoogleTranslator(source='auto', target=intermediate_lang).translate(sentence)
            # Translate back to source language
            back_translated = GoogleTranslator(source='auto', target=src_lang).translate(translated_to_intermediate)
            return back_translated
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Translation failed, retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"Translation failed after {max_retries} attempts. Error: {e}")
                return sentence  # return the original sentence

    return sentence


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
jsonl_file_path = 'Datasets/QA_Pairs/QA_dataset.jsonl'

full_dataset_path = 'Datasets/Augmented/full_dataset.jsonl'
training_file_path = 'Datasets/Augmented/training_dataset.jsonl'
validation_file_path = 'Datasets/Augmented/validation_dataset.jsonl'

# Executes data augmentation and splits the dataset
augment_data(jsonl_file_path, full_dataset_path)
split_data(full_dataset_path, training_file_path, validation_file_path)
