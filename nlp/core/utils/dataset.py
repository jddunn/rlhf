from datasets import load_dataset


class Dataset:
    def __init__(self):
        return

    def load_dataset(self, dataset_name: str = "json", split: str = "train"):
        print("Load dataset: " + dataset_name + ". " + split + "..")
        train_dataset = load_dataset(dataset_name, split=split)
        print("Created train dataset: ", train_dataset)
        return train_dataset
