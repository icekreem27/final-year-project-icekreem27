import json
import nltk
import random
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from tqdm import tqdm
import language_tool_python
from deep_translator import GoogleTranslator


# Ensure necessary NLTK data is downloaded
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

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
    
    valid_words_indices = [(index, (word, get_wordnet_pos(pos))) for index, (word, pos) in enumerate(tagged_words) if word.isalnum() and get_wordnet_pos(pos)]
    if not valid_words_indices:
        return sentence  # no valid words to replace
    
    replace_candidates_indices = random.sample(valid_words_indices, min(n, len(valid_words_indices)))
    
    new_words = words[:]
    for index, (word, pos) in replace_candidates_indices:
        synonyms = get_synonyms_pos(word, pos)
        if synonyms:
            new_words[index] = random.choice(list(synonyms))
    
    return ' '.join(new_words).replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")


def back_translation(sentence, src_lang='en', intermediate_langs=['fr', 'de', 'es']):
    """Perform back-translation on the sentence through a randomly selected intermediate language."""
    intermediate_lang = random.choice(intermediate_langs)
    
    try:
        translated_to_intermediate = GoogleTranslator(source='auto', target=intermediate_lang).translate(sentence)
        back_translated = GoogleTranslator(source='auto', target=src_lang).translate(translated_to_intermediate)
        return back_translated
    except Exception as e:
        return sentence

def correct_grammar(sentences):
    tool = language_tool_python.LanguageTool('en-US')
    corrected_sentences = []
    for sentence in sentences:
        matches = tool.check(sentence)
        corrected_sentence = language_tool_python.utils.correct(sentence, matches)
        corrected_sentences.append(corrected_sentence)
    return corrected_sentences

def augment_data(jsonl_file_path, output_file_path):
    """Augment a dataset with synonym replacement and back-translation."""
    augmented_data = []
        
    # First pass to count lines for the progress bar
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f)

    # Second pass to process the lines with a progress bar
    augmented_data = []
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f)

    sentence_counter = 0  # Initialize a counter to keep track of the sentence number

    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        progress_bar = tqdm(f, total=total_lines, desc="Augmenting data", unit="line")
        for line in progress_bar:
            data = json.loads(line)
            question = data['messages'][1]['content']
            answer = data['messages'][2]['content']

            # Alternate augmentation method based on sentence_counter
            if sentence_counter % 2 == 0:  # Even - Synonym Replacement
                question_aug = correct_grammar(synonym_replacement(question))
                answer_aug = correct_grammar(synonym_replacement(answer))
            else:  # Odd - Back Translation
                question_aug = correct_grammar(back_translation(question))
                answer_aug = correct_grammar(back_translation(answer))
                
                # Check for exact matches with the original before adding
                if question_aug == question:
                    question_aug = None
                if answer_aug == answer:
                    answer_aug = None
            
            # Only add augmented pairs if they're not exact matches
            if question_aug and answer_aug:
                augmented_data.append({"messages": [{"role": "system", "content": data['messages'][0]['content']}, {"role": "user", "content": question_aug}, {"role": "assistant", "content": answer_aug}]})
            
            sentence_counter += 1
    
    # Writes augmented data to file
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(augmented_data, desc="Writing data", unit="item"):
            f_out.write(json.dumps(item) + '\n')

def process_data(data, method):
    if method == "synonym_replacement":
        return synonym_replacement(data)
    elif method == "back_translation":
        return back_translation(data)
    else:
        raise ValueError("Unknown method")
    
    
def augment_and_split_data(input_path, train_path, val_path, val_split=0.2):
    """Augment data and split into training and validation sets."""
    with open(input_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    augmented_lines = []
    for line in tqdm(lines, desc="Augmenting"):
        data = json.loads(line)
        question = data['messages'][1]['content']
        answer = data['messages'][2]['content']

        # Apply augmentation methods alternately
        if len(augmented_lines) % 2 == 0:
            method = "synonym_replacement"
        else:
            method = "back_translation"
        
        question_aug = process_data(question, method)
        answer_aug  = process_data(answer, method)

        # Correct grammar in batches could be implemented here if necessary
        # For simplicity, we're correcting them individually in this example
        question_aug = correct_grammar([question_aug])[0]
        answer_aug = correct_grammar([answer_aug])[0]

        augmented_data = {
            "messages": [
                {"role": "system", "content": data['messages'][0]['content']},
                {"role": "user", "content": question_aug},
                {"role": "assistant", "content": answer_aug}
            ]
        }

        augmented_lines.append(json.dumps(augmented_data))

    # Split data
    random.shuffle(augmented_lines)
    split_index = int(len(augmented_lines) * (1 - val_split))

    training_data = augmented_lines[:split_index]
    validation_data = augmented_lines[split_index:]

    # Write training and validation data to separate files
    with open(train_path, 'w', encoding='utf-8') as f_train:
        f_train.write('\n'.join(training_data))

    with open(val_path, 'w', encoding='utf-8') as f_val:
        f_val.write('\n'.join(validation_data))


# Set file paths for input and output
input_path = 'Datasets/QA_Pairs/QA_dataset.jsonl'
train_path = 'Datasets/Augmented/training_dataset.jsonl'
val_path = 'Datasets/Augmented/validation_dataset.jsonl'

augment_and_split_data(input_path, train_path, val_path)

# will train the model on the full dataset
# then the validation file would be the second time augmented validation_dataset