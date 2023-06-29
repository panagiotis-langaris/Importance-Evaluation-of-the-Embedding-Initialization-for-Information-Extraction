import json
from main import *
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from classes.ADE_Dataset import ADE_Dataset


def my_collate(batch):
    '''
      Custom collate function for the dataloader in
      order to handle the case where the data instances
      have different sizes (no padding).
    '''
    tokens = [item[0] for item in batch]
    ne_tags = [item[1] for item in batch]
    relation_pairs = [item[2] for item in batch]
    embeddings = [item[3] for item in batch]
    ner_tags_numeric = [item[4] for item in batch]
    filename = [item[5] for item in batch]

    return [tokens, ne_tags, relation_pairs, embeddings, ner_tags_numeric, filename]

def my_collate_CLDR_CLNER(batch):
    '''
      Custom collate function for the dataloader in
      order to handle the case where the data instances
      have different sizes (no padding).
    '''
    tokens = [item[0] for item in batch]
    ne_tags = [item[1] for item in batch]
    relation_pairs = [item[2] for item in batch]
    embeddings_NER = [item[3] for item in batch]
    embeddings_RE = [item[4] for item in batch]
    ner_tags_numeric = [item[5] for item in batch]
    filename = [item[6] for item in batch]

    return [tokens, ne_tags, relation_pairs, embeddings_NER, embeddings_RE, ner_tags_numeric, filename]
	
def prepare_dataloaders(lang_model, split_num = 1):
    ### Read the file with the CV splits
    with open(ADE_SPLITS_PATH) as json_file:
        cv_splits = json.load(json_file)

    ### Find the path files for training
    train_files_of_split = []
    if lang_model != 'ORIGINAL_DATASET':
        lm = lang_model + '_'
    else:
        lm = ''

    for f in cv_splits['split_' + str(split_num)]['train set']:
        train_files_of_split.append(lm + f + '.json')
        
    ### Define the lists with the filenames of the test subsets for the corresponding k-fold split
    test_files = []
    for f in cv_splits['split_' + str(split_num)]['test set']:
        test_files.append(lang_model + '_' + f + '.json')

    ### Split the training set files into a training and development/validation subset.
    # Specify the random seed in order to take the same split.
    random.seed(42)
    indexes = list(np.arange(len(train_files_of_split)))
    # 10% of the complete data set
    val_samples_num = int(len(train_files_of_split) * 0.10)
    val_indexes = random.sample(indexes, val_samples_num)
    # Define the lists with the filenames of the training and validation subsets
    train_files = []
    valid_files = []
    for i, f in enumerate(train_files_of_split):
        if i in val_indexes:
            valid_files.append(f)
        else:
            train_files.append(f)

    if lang_model != 'CLDR_CLNER':
        ### Define the data sets objects
        dataset_train = ADE_Dataset(filenames = train_files,
                                    processed_data_path = PROCESSED_DATA_PATH,
                                    folder_path = lang_model + '/')
        dataset_valid = ADE_Dataset(filenames = valid_files,
                                    processed_data_path = PROCESSED_DATA_PATH,
                                    folder_path = lang_model + '/')
        dataset_test = ADE_Dataset(filenames = test_files,
                                   processed_data_path = PROCESSED_DATA_PATH,
                                   folder_path = lang_model + '/')
        
         ### Create the dataloaders
        dataloader_train = DataLoader(dataset_train,
                                      batch_size = hyperparameters['batch_size'],
                                      shuffle = False,
                                      collate_fn = my_collate)

        dataloader_valid = DataLoader(dataset_valid,
                                      batch_size = hyperparameters['batch_size'],
                                      shuffle = False,
                                      collate_fn = my_collate)
        
        dataloader_test = DataLoader(dataset_test,
                                     batch_size = hyperparameters['batch_size'],
                                     shuffle = False,
                                     collate_fn = my_collate)
    
    else:
        ### Define the data sets objects
        dataset_train = ADE_Dataset_CLDR_CLNER(filenames = train_files,
                                               processed_data_path = PROCESSED_DATA_PATH,
                                               folder_path = lang_model + '/split_' + str(split_num) + '/')
        dataset_valid = ADE_Dataset_CLDR_CLNER(filenames = valid_files,
                                               processed_data_path = PROCESSED_DATA_PATH,
                                               folder_path = lang_model + '/split_' + str(split_num) + '/')
        dataset_test = ADE_Dataset_CLDR_CLNER(filenames = test_files,
                                              processed_data_path = PROCESSED_DATA_PATH,
                                              folder_path = lang_model + '/split_' + str(split_num) + '/')
    
        ### Create the dataloaders
        dataloader_train = DataLoader(dataset_train,
                                      batch_size = hyperparameters['batch_size'],
                                      shuffle = False,
                                      collate_fn = my_collate_CLDR_CLNER)

        dataloader_valid = DataLoader(dataset_valid,
                                      batch_size = hyperparameters['batch_size'],
                                      shuffle = False,
                                      collate_fn = my_collate_CLDR_CLNER)
        
        dataloader_test = DataLoader(dataset_test,
                                     batch_size = hyperparameters['batch_size'],
                                     shuffle = False,
                                     collate_fn = my_collate_CLDR_CLNER)

    return dataloader_train, dataloader_valid, dataloader_test
