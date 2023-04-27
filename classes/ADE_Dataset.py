from torch.utils.data import Dataset, DataLoader
import json

class ADE_Dataset(Dataset):
    def __init__(self, filenames, processed_data_path, folder_path):
        super().__init__()
        self.filenames = filenames
        self.processed_data_path = processed_data_path + folder_path

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Load json file
        with open(self.processed_data_path + self.filenames[idx]) as json_file:
            processed_data_file = json.load(json_file)

        # Load the attributes of the dictionary files
        tokens = processed_data_file['tokens']
        ne_tags = processed_data_file['ne tags']
        relation_pairs = processed_data_file['relation pairs']
        embeddings = processed_data_file['embeddings']
        ner_tags_numeric = processed_data_file['ner_tags_numeric']
        filename = str(self.filenames[idx])

        return tokens, ne_tags, relation_pairs, embeddings, ner_tags_numeric, filename