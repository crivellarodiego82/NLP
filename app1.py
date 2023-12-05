import json
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import os
import re

class QASystem:
    def __init__(self, model_name='bert-large-uncased-whole-word-masking-finetuned-squad'):
        # Verifica se il modello è già scaricato localmente
        local_model_path = f'/home/ai/download/directory/{model_name}'
        if os.path.exists(local_model_path):
            print(f"Utilizzo del modello scaricato localmente: {local_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.model = AutoModelForQuestionAnswering.from_pretrained(local_model_path)
        else:
            print(f"Download del modello: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

        self.qa_pipeline = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer)
        self.dataset_file = 'dataset.json'
        self.custom_dataset = self.load_dataset()

    def load_dataset(self):
        try:
            with open(self.dataset_file, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return []

    def save_dataset(self):
        with open(self.dataset_file, 'w') as file:
            json.dump(self.custom_dataset, file, indent=2)

    def train_model(self):
        # Addestra il modello (non necessario in questo caso poiché stiamo usando un modello pre-addestrato)
        pass

    def add_example(self, context, question, answer):
        # Aggiungi un nuovo esempio al dataset
        self.custom_dataset.append({"context": context, "question": question, "answer": answer})
        # Salva il dataset aggiornato
        self.save_dataset()

    def find_context(self, question, dataset):
        # Trova il contesto associato alla domanda nel dataset
        for example in dataset:
            if re.search(re.escape(example['question']), question, re.IGNORECASE):
                return example['context']
        return None

    def answer_question(self, question, context):
        # Esegui Question Answering
        result = self.qa_pipeline(question=question, context=context)
        return result['answer']

# Nome del modello da scaricare
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'

# Directory in cui salvare il modello e il tokenizer
save_directory = '/home/ai/download/directory/'

# Assicurati che la directory esista
os.makedirs(save_directory, exist_ok=True)

# Scarica il tokenizer e il modello localmente
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Salva il tokenizer e il modello nella directory specificata
tokenizer.save_pretrained(os.path.join(save_directory, model_name))
model.save_pretrained(os.path.join(save_directory, model_name))

# Esempio di utilizzo del sistema addestrabile
qa_system = QASystem()

# Addestra il modello (in questo caso, usiamo un dataset di esempio)
qa_system.train_model()

while True:
    # Fornisci una domanda dall'utente
    user_question = input("Inserisci una domanda (o 'exit' per uscire): ")

    # Esci se l'utente digita 'exit'
    if user_question.lower() == 'exit':
        break

    # Trova il contesto associato alla domanda nel dataset
    context = qa_system.find_context(user_question, qa_system.custom_dataset)

    if context is not None:
        # Ottieni la risposta alla domanda
        answer = qa_system.answer_question(user_question, context)
        print("Risposta:", answer)
    else:
        print("Contesto non trovato per la domanda:", user_question)

    # Chiedi all'utente di fornire un nuovo esempio e aggiungilo al dataset
    new_example_context = input("Inserisci il contesto del nuovo esempio: ")
    new_example_question = input("Inserisci la domanda del nuovo esempio: ")
    new_example_answer = input("Inserisci la risposta del nuovo esempio: ")

    qa_system.add_example(new_example_context, new_example_question, new_example_answer)

    print("Nuovo esempio aggiunto al dataset.")
