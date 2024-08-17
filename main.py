from dataset.extractor import Extractor
from dataset.preprocess import make_imbalanced, train_test_split, save, save_undersampling, get_train_df, get_train_df_generated
from encoder.tfidf import TFIDFVectorizer
from encoder.bert import BERTEmbedding
from augmenter.augmenter import Augmenter
from textattack.augmentation import EmbeddingAugmenter, CharSwapAugmenter, EasyDataAugmenter, WordNetAugmenter, DeletionAugmenter
import pandas as pd
from datasets import load_dataset


dataset = load_dataset('cornell-movie-review-data/rotten_tomatoes')
test_df = pd.DataFrame(dataset['test'])
eval_df = pd.DataFrame(dataset['validation'])

from experiment.bert import BERTExperiment
from experiment.bert_lr import BERTLRExperiment
from experiment.tfidf_lr import TFIDFLRExperiment

extractor = Extractor('output')

df = extractor.extract()

target_class_positive = 1 #positive
target_class_negative = 0 #negative



original_df = get_train_df(df)
print(original_df['label'].value_counts())
TFIDFLRExperiment(original_df, test_df, name='original')

generated_df = get_train_df_generated(df, explode=False)
print(generated_df['label'].value_counts())
TFIDFLRExperiment(generated_df, test_df, name='generated')

full_generated_df = get_train_df_generated(df, explode=True)
print(full_generated_df['label'].value_counts())
TFIDFLRExperiment(full_generated_df, test_df, name='generated-full')

positive_imbalanced_df = make_imbalanced(df, target_class_positive, .25)
positive_train_df = get_train_df(positive_imbalanced_df)
print(positive_train_df['label'].value_counts())
TFIDFLRExperiment(positive_train_df, test_df, name='positive-imbalanced')

negative_imbalanced_df = make_imbalanced(df, target_class_negative, .25)
negative_train_df = get_train_df(negative_imbalanced_df)
print(negative_train_df['label'].value_counts())
TFIDFLRExperiment(negative_train_df, test_df, name='negative-imbalanced')

train_df = pd.read_csv('datasets\\positive\\CharSwapAugmenter.csv', sep=';')
print(train_df['label'].value_counts())
TFIDFLRExperiment(train_df, test_df, name='positive-CharSwapAugmenter')

train_df = pd.read_csv('datasets\\positive\\DeletionAugmenter.csv', sep=';')
print(train_df['label'].value_counts())
TFIDFLRExperiment(train_df, test_df, name='positive-DeletionAugmenter')

train_df = pd.read_csv('datasets\\positive\\EasyDataAugmenter.csv', sep=';')
print(train_df['label'].value_counts())
TFIDFLRExperiment(train_df, test_df, name='positive-EasyDataAugmenter')

train_df = pd.read_csv('datasets\\positive\\EmbeddingAugmenter.csv', sep=';')
print(train_df['label'].value_counts())
TFIDFLRExperiment(train_df, test_df, name='positive-EmbeddingAugmenter')

train_df = pd.read_csv('datasets\\positive\\mine.csv', sep=';')
print(train_df['label'].value_counts())
TFIDFLRExperiment(train_df, test_df, name='positive-mine')

train_df = pd.read_csv('datasets\\positive\\undersample.csv', sep=';')
print(train_df['label'].value_counts())
TFIDFLRExperiment(train_df, test_df, name='positive-undersample')

train_df = pd.read_csv('datasets\\positive\\WordNetAugmenter.csv', sep=';')
print(train_df['label'].value_counts())
TFIDFLRExperiment(train_df, test_df, name='positive-WordNetAugmenter')

train_df = pd.read_csv('datasets\\negative\\CharSwapAugmenter.csv', sep=';')
print(train_df['label'].value_counts())
TFIDFLRExperiment(train_df, test_df, name='negative-CharSwapAugmenter')

train_df = pd.read_csv('datasets\\negative\\DeletionAugmenter.csv', sep=';')
print(train_df['label'].value_counts())
TFIDFLRExperiment(train_df, test_df, name='negative-DeletionAugmenter')

train_df = pd.read_csv('datasets\\negative\\EasyDataAugmenter.csv', sep=';')
print(train_df['label'].value_counts())
TFIDFLRExperiment(train_df, test_df, name='negative-EasyDataAugmenter')

train_df = pd.read_csv('datasets\\negative\\EmbeddingAugmenter.csv', sep=';')
print(train_df['label'].value_counts())
TFIDFLRExperiment(train_df, test_df, name='negative-EmbeddingAugmenter')

train_df = pd.read_csv('datasets\\negative\\mine.csv', sep=';')
print(train_df['label'].value_counts())
TFIDFLRExperiment(train_df, test_df, name='negative-mine')

train_df = pd.read_csv('datasets\\negative\\undersample.csv', sep=';')
print(train_df['label'].value_counts())
TFIDFLRExperiment(train_df, test_df, name='negative-undersample')

train_df = pd.read_csv('datasets\\negative\\WordNetAugmenter.csv', sep=';')
print(train_df['label'].value_counts())
TFIDFLRExperiment(train_df, test_df, name='negative-WordNetAugmenter')






# original_df = get_train_df(df)
# print(original_df['label'].value_counts())
# BERTLRExperiment(original_df, test_df, name='original')

# generated_df = get_train_df_generated(df, explode=False)
# print(generated_df['label'].value_counts())
# BERTLRExperiment(generated_df, test_df, name='generated')

# full_generated_df = get_train_df_generated(df, explode=True)
# print(full_generated_df['label'].value_counts())
# BERTLRExperiment(full_generated_df, test_df, name='generated-full')

# positive_imbalanced_df = make_imbalanced(df, target_class_positive, .25)
# positive_train_df = get_train_df(positive_imbalanced_df)
# print(positive_train_df['label'].value_counts())
# BERTLRExperiment(positive_train_df, test_df, name='positive-imbalanced')

# negative_imbalanced_df = make_imbalanced(df, target_class_negative, .25)
# negative_train_df = get_train_df(negative_imbalanced_df)
# print(negative_train_df['label'].value_counts())
# BERTLRExperiment(negative_train_df, test_df, name='negative-imbalanced')

# train_df = pd.read_csv('datasets\\positive\\CharSwapAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTLRExperiment(train_df, test_df, name='positive-CharSwapAugmenter')

# train_df = pd.read_csv('datasets\\positive\\DeletionAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTLRExperiment(train_df, test_df, name='positive-DeletionAugmenter')

# train_df = pd.read_csv('datasets\\positive\\EasyDataAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTLRExperiment(train_df, test_df, name='positive-EasyDataAugmenter')

# train_df = pd.read_csv('datasets\\positive\\EmbeddingAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTLRExperiment(train_df, test_df, name='positive-EmbeddingAugmenter')

# train_df = pd.read_csv('datasets\\positive\\mine.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTLRExperiment(train_df, test_df, name='positive-mine')

# train_df = pd.read_csv('datasets\\positive\\undersample.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTLRExperiment(train_df, test_df, name='positive-undersample')

# train_df = pd.read_csv('datasets\\positive\\WordNetAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTLRExperiment(train_df, test_df, name='positive-WordNetAugmenter')

# train_df = pd.read_csv('datasets\\negative\\CharSwapAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTLRExperiment(train_df, test_df, name='negative-CharSwapAugmenter')

# train_df = pd.read_csv('datasets\\negative\\DeletionAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTLRExperiment(train_df, test_df, name='negative-DeletionAugmenter')

# train_df = pd.read_csv('datasets\\negative\\EasyDataAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTLRExperiment(train_df, test_df, name='negative-EasyDataAugmenter')

# train_df = pd.read_csv('datasets\\negative\\EmbeddingAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTLRExperiment(train_df, test_df, name='negative-EmbeddingAugmenter')

# train_df = pd.read_csv('datasets\\negative\\mine.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTLRExperiment(train_df, test_df, name='negative-mine')

# train_df = pd.read_csv('datasets\\negative\\undersample.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTLRExperiment(train_df, test_df, name='negative-undersample')

# train_df = pd.read_csv('datasets\\negative\\WordNetAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTLRExperiment(train_df, test_df, name='negative-WordNetAugmenter')





# original_df = get_train_df(df)
# print(original_df['label'].value_counts())
# BERTExperiment(original_df, eval_df, test_df, name='original')

# full_generated_df = get_train_df_generated(df, explode=True)
# print(full_generated_df['label'].value_counts())
# BERTExperiment(full_generated_df, eval_df, test_df, name='generated-full')

# generated_df = get_train_df_generated(df, explode=False)
# print(generated_df['label'].value_counts())
# BERTExperiment(generated_df, eval_df, test_df, name='generated')

# positive_imbalanced_df = make_imbalanced(df, target_class_positive, .25)
# positive_train_df = get_train_df(positive_imbalanced_df)
# print(positive_train_df['label'].value_counts())
# BERTExperiment(positive_train_df, eval_df, test_df, name='positive-imbalanced')


# negative_imbalanced_df = make_imbalanced(df, target_class_negative, .25)
# negative_train_df = get_train_df(negative_imbalanced_df)
# print(negative_train_df['label'].value_counts())
# BERTExperiment(negative_train_df, eval_df, test_df, name='negative-imbalanced')

# train_df = pd.read_csv('datasets\\positive\\CharSwapAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTExperiment(train_df, eval_df, test_df, name='positive-CharSwapAugmenter')

# train_df = pd.read_csv('datasets\\positive\\DeletionAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTExperiment(train_df, eval_df, test_df, name='positive-DeletionAugmenter')

# train_df = pd.read_csv('datasets\\positive\\EasyDataAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTExperiment(train_df, eval_df, test_df, name='positive-EasyDataAugmenter')

# train_df = pd.read_csv('datasets\\positive\\EmbeddingAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTExperiment(train_df, eval_df, test_df, name='positive-EmbeddingAugmenter')

# train_df = pd.read_csv('datasets\\positive\\mine.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTExperiment(train_df, eval_df, test_df, name='positive-mine')

# train_df = pd.read_csv('datasets\\positive\\undersample.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTExperiment(train_df, eval_df, test_df, name='positive-undersample')

# train_df = pd.read_csv('datasets\\positive\\WordNetAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTExperiment(train_df, eval_df, test_df, name='positive-WordNetAugmenter')

# train_df = pd.read_csv('datasets\\negative\\CharSwapAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTExperiment(train_df, eval_df, test_df, name='negative-CharSwapAugmenter')

# train_df = pd.read_csv('datasets\\negative\\DeletionAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTExperiment(train_df, eval_df, test_df, name='negative-DeletionAugmenter')

# train_df = pd.read_csv('datasets\\negative\\EasyDataAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTExperiment(train_df, eval_df, test_df, name='negative-EasyDataAugmenter')

# train_df = pd.read_csv('datasets\\negative\\EmbeddingAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTExperiment(train_df, eval_df, test_df, name='negative-EmbeddingAugmenter')

# train_df = pd.read_csv('datasets\\negative\\mine.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTExperiment(train_df, eval_df, test_df, name='negative-mine')

# train_df = pd.read_csv('datasets\\negative\\undersample.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTExperiment(train_df, eval_df, test_df, name='negative-undersample')

# train_df = pd.read_csv('datasets\\negative\\WordNetAugmenter.csv', sep=';')
# print(train_df['label'].value_counts())
# BERTExperiment(train_df, eval_df, test_df, name='negative-WordNetAugmenter')








# extractor = Extractor('output')

# df = extractor.extract()

# target_class = 1 #positive
# target_class = 0 #negative

# imbalanced_df = make_imbalanced(df, target_class, .25)


# # save(imbalanced_df, target_class)
# save_undersampling(imbalanced_df, target_class)


# augmenter = Augmenter(imbalanced_df, target_class, EmbeddingAugmenter(), 'EmbeddingAugmenter')
# augmenter = Augmenter(imbalanced_df, target_class, CharSwapAugmenter(), 'CharSwapAugmenter')
# augmenter = Augmenter(imbalanced_df, target_class, EasyDataAugmenter(), 'EasyDataAugmenter')
# augmenter = Augmenter(imbalanced_df, target_class, WordNetAugmenter(), 'WordNetAugmenter')
# augmenter = Augmenter(imbalanced_df, target_class, DeletionAugmenter(), 'DeletionAugmenter')




