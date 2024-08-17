import pandas as pd
from textattack.augmentation import EmbeddingAugmenter as TextAttackEmbeddingAugmenter
from augmenter.augmenter import Augmenter
from tqdm import tqdm

class EmbeddingAugmenter(Augmenter):
    def __init__(self, df: pd.DataFrame, target_label: int) -> None:
        super().__init__(df)

        dataset = []
        count = 0
        for _, row in df.iterrows():
            count += 1
            print(f'{count}/{len(df)}')
            data = dict()
            data['text'] = row['original']
            data['label'] = row['label']
            
            dataset.append(data)
            
            if data['label'] == target_label:
                generated_text = self.augment(TextAttackEmbeddingAugmenter(), data['text'])
                for text in generated_text:
                    generate_data = dict()
                    generate_data['text'] = text
                    generate_data['label'] = data['label']
                    dataset.append(generate_data)

        df = pd.DataFrame(dataset)
        df.to_csv('EmbeddingAugmenterDataset.csv', sep=';')
        print(df['label'].value_counts())

# # Initialize the augmenter with BERT embeddings
# augmenter = EmbeddingAugmenter()

# # Original sentence
# text = "Text augmentation is a useful technique in natural language processing."

# # Perform augmentation
# augmented_texts = augmenter.augment(text)

# # Display the results
# print("Original Text:", text)
# print("Augmented Texts:", augmented_texts)