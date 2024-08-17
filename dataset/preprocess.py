import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split


def make_imbalanced(df: pd.DataFrame, minority_class: int, ratio: int):
    # Calculate the number of samples per class
    class_counts = df['label'].value_counts()
    
    # Ensure the provided minority class exists in the DataFrame
    if minority_class not in class_counts:
        raise ValueError(f"The minority class '{minority_class}' does not exist in the daset.")
    
    # Identify the majority class
    majority_class = class_counts[class_counts.index != minority_class].idxmax()

    # Calculate the number of samples to keep from the minority class
    majority_count = class_counts[minority_class]
    minority_count = int(majority_count * ratio)
    
    # Sample the minority class
    majority_class_samples = df[df['label'] == majority_class]
    
    # Get all samples from the majority class
    minority_class_samples = df[df['label'] == minority_class].sample(n=minority_count, random_state=42)
    
    # Concatenate the majority and minority class samples to create an imbalanced dataset
    imbalanced_df = pd.concat([majority_class_samples, minority_class_samples], axis=0)

    return imbalanced_df


def train_test_split(df: pd.DataFrame, text_column: str, label_column: str, test_ratio: float):
    # Separate the text and labels
    X = df[text_column]
    y = df[label_column]
    
    # Perform a stratified split
    X_train, X_test, y_train, y_test = sk_train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=42)
    
    # Combine the text and labels back into DataFrames
    train_df = pd.DataFrame({text_column: X_train, label_column: y_train})
    test_df = pd.DataFrame({text_column: X_test, label_column: y_test})
    
    return train_df, test_df


def get_train_df(df: pd.DataFrame):
    train_df = pd.DataFrame()
    train_df['text'] = df['original']
    train_df['label'] = df['label']
    return train_df

def get_train_df_generated(df: pd.DataFrame, explode: bool = False):
    train_df = pd.DataFrame()
    if explode:
        train_df['text'] = df['generated'].explode()
    else:
        train_df['text'] = [texts[0] for texts in df['generated']]
    train_df['label'] = df['label']
    return train_df

def save(df: pd.DataFrame, target_class: int):
    
    dataset = []

    for _, row in df.iterrows():
        data = dict()
        data['text'] = row['original']
        data['label'] = row['label']

        dataset.append(data)

        if data['label'] == target_class:

            for i in range(3):
                generate_data = dict()
                generate_data['text'] = row['generated'][i]
                generate_data['label'] = row['label']
                dataset.append(generate_data)
                
    df = pd.DataFrame(dataset)
    df.to_csv(f'mine.csv', sep=';')
    print(df['label'].value_counts())


from sklearn.utils import resample


def save_undersampling(df: pd.DataFrame, target_class: int):
    # Separate the minority and majority classes

    dataset = []

    for _, row in df.iterrows():
        data = dict()
        data['text'] = row['original']
        data['label'] = row['label']

        dataset.append(data)

                
    df = pd.DataFrame(dataset)

    df_minority = df[df['label'] == target_class]
    df_majority = df[df['label'] != target_class]

    # Perform undersampling on the majority class
    df_majority_undersampled = resample(df_majority,
                                        replace=False,  # sample without replacement
                                        n_samples=len(df_minority),  # match minority class count
                                        random_state=42)  # reproducible results

    # Combine the minority class with the undersampled majority class
    df_undersampled = pd.concat([df_minority, df_majority_undersampled])

    # Shuffle the resulting DataFrame to mix the classes
    # df_undersampled = df_undersampled.sample(frac=1, random_state=42).reset_index(drop=True)

    df_undersampled.to_csv(f'undersample.csv', sep=';')
    print(df_undersampled['label'].value_counts())

    return df_undersampled