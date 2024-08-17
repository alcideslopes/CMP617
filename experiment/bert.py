import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import json

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


class BERTExperiment:
    def __init__(self, train_df: pd.DataFrame, eval_df: pd.DataFrame, test_df: pd.DataFrame, name: str) -> None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Tokenize the input data
        train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=512)
        eval_encodings = tokenizer(eval_df['text'].tolist(), truncation=True, padding=True, max_length=512)
        test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=512)

        # Convert labels to tensor format
        train_labels = torch.tensor(train_df['label'].tolist())
        eval_labels = torch.tensor(eval_df['label'].tolist())
        test_labels = torch.tensor(test_df['label'].tolist())

        # Create datasets
        train_dataset = CustomDataset(train_encodings, train_labels)
        eval_dataset = CustomDataset(eval_encodings, eval_labels)
        test_dataset = CustomDataset(test_encodings, test_labels)

        # Load BERT model for sequence classification
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch"
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)

        # Generate classification report
        y_test = test_labels.numpy()
        report_dict = classification_report(y_test, preds, target_names=['negative', 'positive'], output_dict=True)

        with open(f'{name}.json', 'w') as json_file:
            json.dump(report_dict, json_file, indent=4)

        # cm = confusion_matrix(y_test, preds, labels=[0, 1])

        # with np.errstate(divide='ignore', invalid='ignore'):
        #     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #     cm_normalized[np.isnan(cm_normalized)] = 0  # Set NaNs to zero


        # disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=['negative', 'positive'])
        # disp.plot(cmap=plt.cm.Blues)
        # disp.im_.set_clim(0, 1)
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.show()



      