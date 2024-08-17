import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoModel, AutoTokenizer
from encoder.encoder import Encoder


class BERTEmbedding(Encoder):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer_name = 'google-bert/bert-base-uncased'
        self.model_name = 'google-bert/bert-base-uncased'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_model()


    def _setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

    
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

    
