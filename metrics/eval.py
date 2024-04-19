import json
import tkinter as tk
from tkinter import messagebox, scrolledtext
import sys
import csv
from pathlib import Path
from tkinter import simpledialog

# Get the path of the current file
project_root = Path(__file__).resolve().parent.parent
# Add the project root to the sys.path if not already added
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from Tools.embedding import load_embeddings, ask, ask_no_embeddings

class EvaluationApp:
    def __init__(self, master, df, models):
        self.master = master
        self.df = df
        self.models = models
        self.questions = []
        self.current_question_index = -1
        
        # Collect participant ID
        self.participant_id = simpledialog.askstring("Input", "What is your participant ID?",
                                                      parent=self.master)
        if not self.participant_id:
            self.master.destroy()
            return
        
        # Results list now includes participant ID and model chosen
        self.results = []
        
        # Setting up the window
        self.master.title("User Evaluation")
        self.master.geometry("1200x1000")
        self.master.configure(bg='#f0f0f0')
        
        self.question_label = tk.Label(master, text="", wraplength=760, justify="left", font=('Arial', 20), bg='#f0f0f0', fg='#555')
        self.question_label.pack(pady=(10, 20))
        
        self.answers_frame = tk.Frame(master, bg='#f0f0f0')
        self.answers_frame.pack(fill="both", expand=True, padx=20)
                
        submit_button = tk.Button(self.master, text="Submit", command=self.submit_choice, font=('Arial', 14, 'bold'), bg='#4CAF50', fg='black', padx=20, pady=10)
        submit_button.pack(pady=(20, 10))
        
        self.load_questions()
        self.next_question()
    
    def load_questions(self):
        path = "Datasets/User Evaluation/questions.jsonl"
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    # Attempt to parse the line as JSON
                    data = json.loads(line)
                    # Append the relevant content to the questions list
                    self.questions.append(data['messages'][1]['content'])
                except json.JSONDecodeError:
                    # If an error occurs, skip this line
                    continue

    def next_question(self):
        self.current_question_index += 1
        if self.current_question_index < len(self.questions):
            for widget in self.answers_frame.winfo_children():
                widget.destroy()
            self.display_question(self.questions[self.current_question_index])
        else:
            self.finish_evaluation()

    def display_question(self, question):
        answers = []
        self.question_label.config(text=question, pady=35, font=('Arial', 20, 'bold'), justify="center")
        for model in self.models:
            if model != 'base':
                answers.append(ask(question, self.df, model=model))
            else:
                answers.append(ask_no_embeddings(question))
            
        self.choice_var = tk.IntVar(value=-1)
        for idx, answer in enumerate(answers):
            frame = tk.Frame(self.answers_frame, bg='#f0f0f0')
            frame.pack(fill="x", expand=True, pady=5)

            answer_box = tk.Text(frame, wrap=tk.WORD, font=('Arial', 18), borderwidth=2, relief="solid", padx=10, pady=10)
            answer_box.insert(tk.END, answer)
            answer_box.config(state="disabled", bg="#fff", fg='#444')
            answer_box.pack(side="left", fill="both", expand=True, padx=(0, 10))
            answer_box.config(height=8)
            tk.Radiobutton(frame, text="Select", variable=self.choice_var, value=idx, font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='#333', selectcolor="#ddd").pack(side="left", pady=(0, 10))
            

    def submit_choice(self):
        choice = self.choice_var.get()
        if choice >= 0:
            selected_model = self.models[choice]
            self.results.append({
                "participant_id": self.participant_id,
                "question_number": self.current_question_index + 1,
                "model": selected_model
            })
            self.next_question()

    def finish_evaluation(self):
        fieldnames = ['participant_id', 'question_number', 'model']
        with open("Datasets/User Evaluation/evaluation_results.csv", "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerows(self.results)
        self.master.destroy()

if __name__ == "__main__":
    embeddings_path = "Datasets/embeddings.csv"
    models = ['gpt-3.5-turbo', 'ft:gpt-3.5-turbo-0125:personal::90bpxw0O', 'base']

    df = load_embeddings(embeddings_path)
    
    root = tk.Tk()
    app = EvaluationApp(root, df, models)
    root.mainloop()
