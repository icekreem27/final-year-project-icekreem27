import json
import nltk
import random
from nltk.corpus import wordnet
from googletrans import Translator, LANGUAGES
from tqdm import tqdm

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Get synonyms for any given word
def get_synonyms(word):
    synonyms = set()
    # Iterate in WordNet
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # Get the synonym and preprocess it
            synonym = lemma.name().replace('_', ' ').replace('-', ' ').lower()
            synonym = "".join([char for char in synonym if char.isalnum() or char == ' '])
            # Add the synonym to the set
            synonyms.add(synonym)
    # Then remove the original word from the set of synonyms
    if word in synonyms:
        synonyms.remove(word)
    # Return as a list
    return list(synonyms)

# Synonym replacement for any given sentence
def synonym_replacement(sentence, n=2):
    words = nltk.word_tokenize(sentence)  # Tokenize
    new_words = words.copy()  # Create a copy of the tokenized words
    random_word_list = list(set([word for word in words if word.isalnum()]))  # Get unique words
    random.shuffle(random_word_list)  # Shuffle the list of words
    num_replaced = 0
    
    # Iterate over the random word list
    for random_word in random_word_list:
        # Get synonyms for each random word
        synonyms = get_synonyms(random_word)
        # if there are synonyms available
        if len(synonyms) >= 1:
             # Choose a random synonym
            synonym = random.choice(list(synonyms))
            # Replace the random word with the synonym from the list
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:  # Check if the number of replacements is reached
            break
    # return new words list
    return ' '.join(new_words)


def back_translation(sentence, src_lang='en', intermediate_lang='fr'):
    translator = Translator()
    # Translate from source to language (french)
    translated = translator.translate(sentence, src=src_lang, dest=intermediate_lang).text
    # Translate back to source language
    back_translated = translator.translate(translated, src=intermediate_lang, dest=src_lang).text
    return back_translated


def augment_data(jsonl_file_path, output_file_path):
    augmented_data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # Read lines to count
    # Show the progress when looping
    for line in tqdm(lines, desc="Augmenting data", unit="lines"):
        data = json.loads(line)
        question = data['messages'][1]['content']
        answer = data['messages'][2]['content']
        
        # Perform synonym replacement
        syn_question = synonym_replacement(question)
        syn_answer = synonym_replacement(answer)
        
        # Perform back translation
        bt_question = back_translation(question)
        bt_answer = back_translation(answer)
        
        augmented_data.extend([
            data,  # The original data
            {"messages": [{"role": "system", "content": "You are a factual chatbot that will help student learn about the COMP2121 Data Mining module"}, {"role": "user", "content": syn_question}, {"role": "assistant", "content": syn_answer}]},
            {"messages": [{"role": "system", "content": "You are a factual chatbot that will help student learn about the COMP2121 Data Mining module"}, {"role": "user", "content": bt_question}, {"role": "assistant", "content": bt_answer}]}
        ])
    
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(augmented_data, desc="Writing data", unit="items"):
            f_out.write(json.dumps(item) + '\n')

# File paths
jsonl_file_path = 'dataset.jsonl'
output_file_path = 'augmented_dataset.jsonl'

# Run the augmentation
augment_data(jsonl_file_path, output_file_path)
