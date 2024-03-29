import spacy
import json
import nltk
import random
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from torch import mode
from tqdm import tqdm
import language_tool_python
from deep_translator import GoogleTranslator

from openai import OpenAI
client = OpenAI()

# Ensure necessary NLTK data is downloaded
nlp = spacy.load("en_core_web_sm")
tool = language_tool_python.LanguageTool('en-US')

def get_spacy_pos(spacy_token):
    """Converts spaCy POS tags to WordNet POS tags."""
    if spacy_token.pos_ == 'ADJ':
        return wn.ADJ
    elif spacy_token.pos_ == 'VERB':
        return wn.VERB
    elif spacy_token.pos_ == 'NOUN':
        return wn.NOUN
    elif spacy_token.pos_ == 'ADV':
        return wn.ADV
    else:
        return None

def get_best_synonym(token):
    """Get the best synonym for a given spaCy token excluding the word itself."""
    pos = get_spacy_pos(token)  # Use the previously defined function to convert POS
    best_synonym = None
    highest_similarity = 0.0
    
    if pos is not None:
        original_synsets = wn.synsets(token.text, pos=pos)
        if original_synsets:  # Ensure there is at least one synset
            for syn in wn.synsets(token.text, pos=pos):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ').replace('-', ' ').lower()
                    if synonym != token.text.lower():  # Compare with token's lowercase text
                        for orig_syn in original_synsets:
                            similarity = orig_syn.path_similarity(syn)
                            if similarity and similarity > highest_similarity:
                                best_synonym = synonym
                                highest_similarity = similarity
    return best_synonym

def synonym_replacement(sentence, n=3):
    doc = nlp(sentence)
    valid_tokens = [(i, token) for i, token in enumerate(doc) if token.is_alpha and not token.is_stop and get_spacy_pos(token) and token.pos_ != 'PROPN']
    
    if not valid_tokens:
        return sentence  # No valid tokens to replace
    
    replace_candidates = random.sample(valid_tokens, min(n, len(valid_tokens)))
    
    new_tokens = [token.text for token in doc]
    
    for index, token in replace_candidates:
        best_synonym = get_best_synonym(token)
        if best_synonym:
            new_tokens[index] = best_synonym
    
    new_sentence = ' '.join(new_tokens)
    return new_sentence

def back_translation(sentence, src_lang='en', intermediate_langs=['tt', 'da', 'ml']):
    """Perform back-translation on the sentence through a randomly selected intermediate language."""
    intermediate_lang = random.choice(intermediate_langs)
    try:
        translated_to_intermediate = GoogleTranslator(source='auto', target=intermediate_lang).translate(sentence)
        back_translated = GoogleTranslator(source='auto', target=src_lang).translate(translated_to_intermediate)
        return back_translated
    except Exception as e:
        return sentence

def correct_grammar(sentence, tool=tool):
    matches = tool.check(sentence)
    corrected_sentence = language_tool_python.utils.correct(sentence, matches)
    return corrected_sentence

def process_data(data, method):
    if method == "synonym_replacement":
        return synonym_replacement(data)
    elif method == "back_translation":
        return back_translation(data)
    elif method == "paraphrase":
        return paraphrase(data)
    elif method == "gptReplace":
        return gptReplace(data)
    else:
        raise ValueError("Unknown method")
    
def paraphrase(sentence, model="gpt-3.5-turbo"):
    try:
        response = client.chat.completions.create(
            model=model, # model ID
            messages=[
                {"role": "system", "content": "You will paraphrase the following sentence while keeping the original meaning. Maintaining correct context is the highest priority. Use natural language that matches the tone of the original sentence."},
                {"role": "user", "content": sentence}
            ],
            n=1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def gptReplace(sentence, model="gpt-3.5-turbo"):
    try:
        response = client.chat.completions.create(
            model=model, # model ID
            messages=[
                {"role": "system", "content": "You will replace two to three words from the sentence with their synonyms while keeping the original meaning. Maintaining correct context is the highest priority. Use natural language that matches the tone of the original sentence. Do not choose to swap words that are proper nouns."},
                {"role": "user", "content": sentence}
            ],
            n=1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def augment_data(input_path, output_path):
    """Augment data without splitting."""
    with open(input_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    augmented_lines = []
    for line in tqdm(lines, desc="Augmenting"):
        data = json.loads(line)
        question = data['messages'][1]['content']
        answer = data['messages'][2]['content']

        # Apply augmentation methods alternately
        method = "gptReplace" if len(augmented_lines) % 2 == 0 else "paraphrase"
        question_aug = process_data(question, method)
        answer_aug = process_data(answer, method)

        augmented_lines.append({
            "messages": [
                {"role": "system", "content": data['messages'][0]['content']},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })
        
        augmented_lines.append({
            "messages": [
                {"role": "system", "content": data['messages'][0]['content']},
                {"role": "user", "content": question_aug},
                {"role": "assistant", "content": answer_aug}
            ]
        })

    # # Correct grammar in augmented lines
    # for i, data in enumerate(tqdm(augmented_lines, desc="Correcting Grammar")):
    #     question_aug = data['messages'][1]['content']
    #     answer_aug = data['messages'][2]['content']

    #     corrected_question = correct_grammar(question_aug)
    #     corrected_answer = correct_grammar(answer_aug)

    #     # Update sentences if grammar correction changed them
    #     if corrected_question != question_aug or corrected_answer != answer_aug:
    #         data['messages'][1]['content'] = corrected_question
    #         data['messages'][2]['content'] = corrected_answer
    #         augmented_lines[i] = data  # Updated with corrected grammar
    
    # Write augmented data to file
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(augmented_lines, desc="Writing Augmented Data"):
            f_out.write(json.dumps(item) + '\n')
            
def augment_data_no_original(input_path, output_path):
    """Augment data without including original data"""
    with open(input_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    augmented_lines = []
    for line in tqdm(lines, desc="Augmenting"):
        data = json.loads(line)
        question = data['messages'][1]['content']
        answer = data['messages'][2]['content']

        # Apply augmentation methods alternately
        method = "gptReplace" if len(augmented_lines) % 2 == 0 else "paraphrase"
        question_aug = process_data(question, method)
        answer_aug = process_data(answer, method)

        augmented_lines.append({
            "messages": [
                {"role": "system", "content": data['messages'][0]['content']},
                {"role": "user", "content": question_aug},
                {"role": "assistant", "content": answer_aug}
            ]
        })

    # Correct grammar in augmented lines
    # for i, data in enumerate(tqdm(augmented_lines, desc="Correcting Grammar")):
    #     question_aug = data['messages'][1]['content']
    #     answer_aug = data['messages'][2]['content']

    #     corrected_question = correct_grammar(question_aug)
    #     corrected_answer = correct_grammar(answer_aug)

    #     # Update sentences if grammar correction changed them
    #     if corrected_question != question_aug or corrected_answer != answer_aug:
    #         data['messages'][1]['content'] = corrected_question
    #         data['messages'][2]['content'] = corrected_answer
    #         augmented_lines[i] = data  # Updated with corrected grammar
    
    # Write augmented data to file
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(augmented_lines, desc="Writing Augmented Data"):
            f_out.write(json.dumps(item) + '\n')

def split_data(input_path, train_path, val_path, val_split):
    """Split data into training and validation sets."""
    with open(input_path, 'r', encoding='utf-8') as file:
        # Assume each line in the file is a separate JSON object
        lines = [json.loads(line) for line in file]

    # Shuffle the data randomly
    random.shuffle(lines)
    
    # Calculate the index at which to split the data
    split_index = int(len(lines) * (1 - val_split))

    # Split the data into training and validation sets
    training_data = lines[:split_index]
    validation_data = lines[split_index:]

    # Write training data to the training file
    with open(train_path, 'w', encoding='utf-8') as f_train:
        for item in training_data:
            f_train.write(json.dumps(item) + '\n')

    # Write validation data to the validation file
    with open(val_path, 'w', encoding='utf-8') as f_val:
        for item in validation_data:
            f_val.write(json.dumps(item) + '\n')


# Set file paths for input and output
input_path = 'Datasets/QA_Pairs/QA_dataset.jsonl'
train_path = 'Datasets/Augmented/training_dataset.jsonl'
val_path = 'Datasets/Augmented/validation_dataset.jsonl'
full_path = 'Datasets/Augmented/full_dataset.jsonl'

# Dataset split ratio
split = 0.2

# augment_data(input_path, full_path)
# augment_data(full_path, full_path)
split_data(full_path, train_path, val_path, split)
augment_data_no_original(val_path, val_path)

# will train the model on the full dataset
# then the validation file would be the second time augmented validation_dataset