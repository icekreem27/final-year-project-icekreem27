from openai import OpenAI
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Function to call the custom GPT-3.5 model
def generate_text(prompt, model_name):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a factual chatbot that will help student learn about the COMP2121 Data Mining module."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# Function to calculate BLEU score
def calculate_bleu(reference_texts, generated_text):
    # Tokenize the reference texts and the generated text
    reference_tokens = [word_tokenize(text.lower()) for text in reference_texts]
    generated_tokens = word_tokenize(generated_text.lower())
    
    # Use a smoothing function
    chencherry = SmoothingFunction()
    
    # Calculate the BLEU score with smoothing
    bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=chencherry.method1)
    return bleu_score

def calculate_meteor_score(reference_texts, generated_text):
    # METEOR score calculation
    references = [word_tokenize(ref.lower()) for ref in reference_texts]
    hypothesis = word_tokenize(generated_text.lower())
    
    # Calculate METEOR score using NLTK
    score = meteor_score(references, hypothesis)
    return score


if __name__ == "__main__":
    prompt = "What is the instructor stressing in terms of  monitoring other things in the lecture?"
    generated_text = generate_text(prompt, model_name='ft:gpt-3.5-turbo-0125:personal:fulldataset-0-05lr:97r4XJWr')
    print("Generated Text:", generated_text)
    
    reference_texts = [
        "The speaker is stressing the significance of having interfaces to different other IT systems besides monitoring other things.",
        "The reader is underline the importance of having interfaces to various other IT systems in addition to retain track of other things.",
        "The reader is emphasizing the significance of having interfaces to different other IT systems along with keeping track of other things."
    ]
    
    bleu_score = calculate_bleu(reference_texts, generated_text)
    print("BLEU Score:", bleu_score)

    meteor_score = calculate_meteor_score(reference_texts, generated_text)
    print(f"METEOR Score:", meteor_score)
