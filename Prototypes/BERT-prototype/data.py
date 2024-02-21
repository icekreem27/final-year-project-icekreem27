import json
import random
from googletrans import Translator
from nltk.corpus import wordnet  # Import NLTK's WordNet interface for synonym replacement

# Load your SQuAD JSON file
with open('augmented_squad_dataset_back_translation.json', 'r') as json_file:
    squad_data = json.load(json_file)

# Initialize the translator
translator = Translator()

# Define a list of language codes for back translation
languages = ['fr', 'es', 'de', 'it']  # Example: French, Spanish, German, Italian, etc.

# Define a function for synonym replacement
def synonym_replace(text):
    words = text.split()
    augmented_words = []

    for word in words:
        # Randomly decide whether to replace the word with a synonym (50% chance)
        if random.random() < 0.5:
            # Get synonyms for the word using WordNet
            synonyms = wordnet.synsets(word)
            if synonyms:
                synonym = random.choice(synonyms).lemmas()[0].name()  # Select a random synonym
                augmented_words.append(synonym)
            else:
                augmented_words.append(word)
        else:
            augmented_words.append(word)

    return ' '.join(augmented_words)

# Define a function for data augmentation using back translation and synonym replacement
def augment_data(squad_data):
    augmented_data = []

    for example in squad_data['data']:
        augmented_paragraphs = []

        for paragraph in example['paragraphs']:
            augmented_qas = []

            for qa in paragraph['qas']:
                question = qa['question']

                # Select random source and target languages from the list for back translation
                src_lang = 'en'  # Source language (English)
                dest_lang = random.choice(languages)  # Target language

                # Translate the question to the target language and back
                translated_question = translator.translate(question, src=src_lang, dest=dest_lang).text
                back_translated_question = translator.translate(translated_question, src=dest_lang, dest=src_lang).text

                # Apply synonym replacement to the back-translated question
                augmented_question = synonym_replace(back_translated_question)

                # Create a new question with augmented text and keep the original question
                augmented_qa = {
                    'question': augmented_question,
                    'original_question': question,  # Keep the original question for reference
                    'id': qa['id'],
                    'answers': qa['answers']
                }

                augmented_qas.append(augmented_qa)

            augmented_paragraph = {
                'context': paragraph['context'],
                'qas': augmented_qas
            }

            augmented_paragraphs.append(augmented_paragraph)

        augmented_example = {
            'title': example['title'],
            'paragraphs': augmented_paragraphs
        }

        augmented_data.append(augmented_example)

    return augmented_data

# Augment the data using back translation, synonym replacement, and keeping original text
augmented_data = augment_data(squad_data)

# Save the augmented data to a new JSON file
with open('augmented_squad_dataset_synonym_back_translation.json', 'w') as output_file:
    augmented_squad_data = {
        'version': squad_data['version'],
        'data': augmented_data
    }
    json.dump(augmented_squad_data, output_file)
