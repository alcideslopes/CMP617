import pandas as pd
from textattack.augmentation import Augmenter as TextAttackAugmenter

class Augmenter:
    def __init__(self, df: pd.DataFrame, target_label: int, augmenter: TextAttackAugmenter, name: str) -> None:
        self.df = df
        self.augmenter = augmenter
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
                generated_text = self.generate_multiple_augmentations(data['text'])
                for text in generated_text:
                    generate_data = dict()
                    generate_data['text'] = text
                    generate_data['label'] = data['label']
                    dataset.append(generate_data)

        df = pd.DataFrame(dataset)
        df.to_csv(f'{name}.csv', sep=';')
        print(df['label'].value_counts())


    def generate_multiple_augmentations(self, text, num_augmentations=3):
        augmented_texts = list()  # Use a set to avoid duplicates

        augmented_text = self.augment(text)
        augmented_texts.append(augmented_text[0])  # Augment returns a list, so take the first item

        for i in range(num_augmentations - 1):
            augmented_text = self.augment(augmented_texts[i])
            augmented_texts.append(augmented_text[0])  # Augment returns a list, so take the first item

        return augmented_texts
    
    def augment(self, text: str) -> str:
        return self.augmenter.augment(text)
    

