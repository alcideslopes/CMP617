from datasets import load_dataset


class Dataset:

    def __init__(self, dataset_name: str) -> None:
        dataset = load_dataset(dataset_name)

        self.train_dataset = dataset['train']
        self.validation_dataset = dataset['validation']
        self.test_dataset = dataset['test']


