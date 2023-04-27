import torch
from helper_functions.get_chunks import *

# Import module from parent directory
import sys
sys.path.append('C:/Users/plang/Desktop/2. Leuven/Thesis/1. Code')
from common_config import *


def prepare_inputs(token_ner_tags, token_embeddings):

    ### Initialize the embedding for the predicted/learned BIO tags of the NER model and store the entity indexes for RC
    tag_n_token_embeddings = torch.tensor([], device='cuda')

    # Get the chunks of the entities
    chunks = get_chunks(token_ner_tags.tolist(), mapping_ne_tags) # result = [("entity", idx_start, idx_end), (...), (...)]

    # Initialize lists with indices of entities for potential pairs
    drug_entity_indexes = []
    effect_entity_indexes = []
    for i in range(len(chunks)):
        if chunks[i][0]=='DRUG':
            drug_entity_indexes.append(chunks[i][2]) # Get the ending index of the entity
        elif chunks[i][0]=='AE':
            effect_entity_indexes.append(chunks[i][2]) # Get the ending index of the entity

    for i in range(len(token_ner_tags)):
        # Concatenate the appropriate one-hot tag embedding for each token
        if token_ner_tags[i] == 0:
            one_hot_tag = torch.tensor([1, 0, 0, 0, 0], device='cuda') # One hot encoded embedding for NER tag 'O'
        elif token_ner_tags[i] == 1:
            one_hot_tag = torch.tensor([0, 1, 0, 0, 0], device='cuda') # One hot encoded embedding for NER tag 'B-DRUG'
        elif token_ner_tags[i] == 2:
            one_hot_tag = torch.tensor([0, 0, 1, 0, 0], device='cuda') # One hot encoded embedding for NER tag 'I-DRUG'
        elif token_ner_tags[i] == 3:
            one_hot_tag = torch.tensor([0, 0, 0, 1, 0], device='cuda') # One hot encoded embedding for NER tag 'B-AE'
        elif token_ner_tags[i] == 4:
            one_hot_tag = torch.tensor([0, 0, 0, 0, 1], device='cuda') # One hot encoded embedding for NER tag 'B-DRUG'

        tmp = torch.cat((one_hot_tag, token_embeddings[i]), dim=0)
        tmp.unsqueeze_(0)
        tag_n_token_embeddings = torch.cat((tag_n_token_embeddings, tmp), dim=0)

    return tag_n_token_embeddings, drug_entity_indexes, effect_entity_indexes