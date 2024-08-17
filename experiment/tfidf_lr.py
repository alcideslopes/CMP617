import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression



class TFIDFLRExperiment:

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, name: str) -> None:
        vectorizer = TfidfVectorizer()

        train_encodings = vectorizer.fit_transform(train_df['text'].tolist())
        test_encodings = vectorizer.transform(test_df['text'].tolist())

        classifier = LogisticRegression()

        classifier.fit(train_encodings, train_df['label'])

        # Make predictions
        y_pred = classifier.predict(test_encodings)


        y_pred = np.vectorize(self.convert)(y_pred)
        y_test = np.vectorize(self.convert)(test_df['label'])


        report_dict = classification_report(y_test, y_pred, target_names=['negative', 'positive'], output_dict=True)

        with open(f'{name}.json', 'w') as json_file:
            json.dump(report_dict, json_file, indent=4)


    def convert(self, value):
        return 'positive' if value == 1 else 'negative'
    
    

    


      