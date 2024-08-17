from sklearn.feature_extraction.text import TfidfVectorizer
from encoder.encoder import Encoder

class TFIDFVectorizer(Encoder):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, texts):
        vectorizer = TfidfVectorizer()
        encoded_text = vectorizer.fit_transform(texts)
        return encoded_text