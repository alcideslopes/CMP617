import re
import os
import json
import pandas as pd

class Extractor:
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.extract()


    def extract(self):
        # Get list of all files in the directory
        files = os.listdir(self.root_dir)

        generated_texts = []
        generated_labels = []
        original_texts = []
        original_labels = []

        # Filter out only the JSON files
        json_files = [file for file in files if file.endswith('.json')]

        # Read each JSON file and convert it to a Python dictionary
        for json_file in json_files:
            file_path = os.path.join(self.root_dir, json_file)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                phrases = self.preprocess_text(data['generated_text'])
                label = data['label']

                real_text = data['original_text']
                real_label = label
                original_texts.append(real_text)
                original_labels.append(real_label)

                # generated_texts.append(phrases[0])
                # generated_labels.append(label)

                for phrase in phrases:
                    generated_texts.append(phrase)
                    generated_labels.append(label)

        generated_df = pd.DataFrame({'text': generated_texts, 'label': generated_labels})
        original_df = pd.DataFrame({'text': original_texts, 'label': original_labels})

        return original_df, generated_df
    


    def preprocess_text(self, text: str) -> list:
        cleaned_text = re.sub(r'\d+\.\s', '', text)

        phrases = cleaned_text.split('\n')

        return phrases
