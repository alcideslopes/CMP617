import re
import os
import json
import pandas as pd

class Extractor:
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir

    def extract(self):
        # Get list of all files in the directory
        files = os.listdir(self.root_dir)

        dataset = []

        # Filter out only the JSON files
        json_files = [file for file in files if file.endswith('.json')]

        # Read each JSON file and convert it to a Python dictionary
        for json_file in json_files:
            file_path = os.path.join(self.root_dir, json_file)
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)

                data = dict()
                data['original'] = json_data['original_text']
                data['generated'] = self.preprocess_text(json_data['generated_text'])
                data['label'] = json_data['label']

                dataset.append(data)

        return pd.DataFrame(dataset)
    
    
    def preprocess_text(self, text: str) -> list:
        cleaned_text = re.sub(r'\d+\.\s', '', text)

        phrases = cleaned_text.split('\n')

        return phrases


