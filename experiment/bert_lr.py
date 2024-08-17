import pandas as pd
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression



class BERTLRExperiment:

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, name: str) -> None:
        self.tokenizer_name = 'google-bert/bert-base-uncased'
        self.model_name = 'google-bert/bert-base-uncased'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_model()

        train_encodings = self.encode(train_df['text'].tolist())
        test_encodings = self.encode(test_df['text'].tolist())

        classifier = LogisticRegression()

        classifier.fit(train_encodings, train_df['label'])

        # Make predictions
        y_pred = classifier.predict(test_encodings)


        y_pred = np.vectorize(self.convert)(y_pred)
        y_test = np.vectorize(self.convert)(test_df['label'])


        report_dict = classification_report(y_test, y_pred, target_names=['negative', 'positive'], output_dict=True)

        with open(f'{name}.json', 'w') as json_file:
            json.dump(report_dict, json_file, indent=4)


    def _setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

    def convert(self, value):
        return 'positive' if value == 1 else 'negative'
    
    def encode(self, texts):

        encoded_texts = []

        for text in tqdm(texts):
            tokenized_text = self.tokenizer(text, add_special_tokens=True,
                                                padding=True,
                                                truncation=True,
                                                return_attention_mask=True, 
                                                return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.model(**tokenized_text)
                sentence_embedding = outputs.last_hidden_state.mean(dim=1)[0] 
                encoded_texts.append(np.asarray(sentence_embedding.cpu()))

        return encoded_texts

    # def __init__(self, train_df: pd.DataFrame, eval_df: pd.DataFrame, test_df: pd.DataFrame, name: str) -> None:
    #     # Tokenize the input data
    #     train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=512)
    #     eval_encodings = tokenizer(eval_df['text'].tolist(), truncation=True, padding=True, max_length=512)
    #     test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=512)

        

    #     # Evaluate the model
    #     predictions = trainer.predict(test_dataset)
    #     preds = np.argmax(predictions.predictions, axis=-1)

    #     # Generate classification report
    #     y_test = test_labels.numpy()
    #     report_dict = classification_report(y_test, preds, target_names=['negative', 'positive'], output_dict=True)

    #     with open(f'{name}.json', 'w') as json_file:
    #         json.dump(report_dict, json_file, indent=4)

    #     # cm = confusion_matrix(y_test, preds, labels=[0, 1])

    #     # with np.errstate(divide='ignore', invalid='ignore'):
    #     #     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     #     cm_normalized[np.isnan(cm_normalized)] = 0  # Set NaNs to zero


    #     # disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=['negative', 'positive'])
    #     # disp.plot(cmap=plt.cm.Blues)
    #     # disp.im_.set_clim(0, 1)
    #     # plt.xticks(rotation=45)
    #     # plt.tight_layout()
    #     # plt.show()



      