import json
import csv
from pathlib import Path
import sys

# Get the path of the current file
project_root = Path(__file__).resolve().parent.parent
# Add the project root to the sys.path if not already added
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from Tools.embedding import load_embeddings, ask, ask_no_embeddings

class AutoEvaluation:
    def __init__(self, df, models):
        self.df = df
        self.models = models
        self.questions = []
        self.answers = []
        self.load_questions()
        self.evaluate_all_questions()

    def load_questions(self):
        path = "Datasets/User Evaluation/questions.jsonl"
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    self.questions.append(data['messages'][1]['content'])
                except json.JSONDecodeError:
                    continue

    def evaluate_all_questions(self):
        for question in self.questions:
            results = []
            for model in self.models:
                if model != 'base':
                    answer = ask(question, self.df, model=model)
                else:
                    answer = ask_no_embeddings(question)
                results.append({"question": question, "model": model, "answer": answer})
            self.answers.append(results)
        self.save_results()

    def save_results(self):
        fieldnames = ['question', 'model', 'answer']
        with open("Datasets/User Evaluation/auto_evaluation_results.csv", "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            for question_answers in self.answers:
                writer.writerows(question_answers)

if __name__ == "__main__":
    embeddings_path = "Datasets/embeddings.csv"
    models = ['gpt-3.5-turbo', 'ft:gpt-3.5-turbo-0125:personal::90bpxw0O', 'base']
    df = load_embeddings(embeddings_path)
    auto_evaluator = AutoEvaluation(df, models)
