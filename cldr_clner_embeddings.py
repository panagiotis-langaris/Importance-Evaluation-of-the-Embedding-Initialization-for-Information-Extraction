import configparser
import json
import numpy as np
import random 
import torch
import os
import sys
from utils.character_cnn import CharacterIndexer
from modeling.character_bert import CharacterBertModel
from common_config import *
#%cd C:\Users\plang\anaconda3\envs\character-bert # Change path to run character-bert
from transformers import BertTokenizer
from classes.ADE_Dataset import ADE_Dataset
import sys
from sys import exit

if len(sys.argv) > 0:
    SPLIT_NUM = int(sys.argv[1])
else:
    print('Error. Missing parameter of k-fold number.')
    print('Program terminated.')
    exit()

CHARACTER_BERT_PATH = 'C:/Users/plang/Desktop/2. Leuven/Thesis/1. Code/character-bert-main/'

SAVED_MODEL_RE_PATH = 'C:/Users/plang/Desktop/2. Leuven/Thesis/1. Code/CLDR_CLNER_models/CLDR_CLNER_weights/CLDR/'
SAVED_MODEL_RE_PATH = SAVED_MODEL_RE_PATH + '/split_' + str(SPLIT_NUM) + '/best_val_trained_model.pt'

SAVED_MODEL_NER_PATH = 'C:/Users/plang/Desktop/2. Leuven/Thesis/1. Code/CLDR_CLNER_models/CLDR_CLNER_weights/CLNER/'
SAVED_MODEL_NER_PATH = SAVED_MODEL_NER_PATH + '/split_' + str(SPLIT_NUM) + '/best_val_trained_model.pt'

PATH_OUT = 'C:/Users/plang/Desktop/2. Leuven/Thesis/1. Code/dataset/processed_data/CLDR_CLNER'

class Model_NE(torch.nn.Module):
	def __init__(self, dropout_ne, characterBERT_path):
		super(Model_NE, self).__init__()

		# Text encoder part
		self.characterBERT_path = characterBERT_path + '/pretrained-models/medical_character_bert/'
		self.characterBERT = CharacterBertModel.from_pretrained(self.characterBERT_path)

		# Freeze the first 6 encoding layers and the initial embedding layer
		#modules = [self.characterBERT.embeddings, *self.characterBERT.encoder.layer[:6]]
		#for module in modules:
		#	for param in module.parameters():
		#		param.requires_grad = False

		# NE representation part
		self.fc1_ne = torch.nn.Linear(768, 768)
		self.drop_ne_layer = torch.nn.Dropout(dropout_ne)

	def forward(self, tokens):
		'''
			sent_id: the encoded sentence (containing [CLS], [SEP] and [PAD] tokens)
			mask: the masking of the sentence (indication of true or padded token)
		'''
		output = self.characterBERT(tokens)

		# Pass each token representation from a FC layer
		ne_rep = self.drop_ne_layer(output[0])
		ne_rep = self.fc1_ne(ne_rep)

		return ne_rep[0]


class CLDR_CLNER_Pre_Proc:
    def __init__(self, processed_data_path, lang_model):
        self.processed_data_path = processed_data_path
        self.lang_model = lang_model
        self.out_folder_path = lang_model + '/' + 'split_' + str(SPLIT_NUM) + '/'
        self.characterBERT_path = CHARACTER_BERT_PATH

        # Find the path files for training
        self.processed_files_list = []
        for file_index in range(hyperparameters['dataset_length']):
            self.processed_files_list.append(str(file_index) + '.json')

        self.original_dataset = ADE_Dataset(filenames = self.processed_files_list,
                                            processed_data_path = processed_data_path,
                                            folder_path = 'original/')
   
    def save_json(self, tokens, ne_tags, relation_pairs, out_text_encoder_NER, extracted_embeddings_ready_RE, ner_tags_numeric, train_file, lang_model_name):

        dict_out = {'tokens': tokens,
                    'ne tags': ne_tags,
                    'relation pairs': relation_pairs,
                    'embeddings_NER': out_text_encoder_NER,
                    'embeddings_RE': extracted_embeddings_ready_RE,
                    'ner_tags_numeric': ner_tags_numeric
                    }

        f_name = lang_model_name + train_file

        lm_output_path = self.processed_data_path + self.out_folder_path
        if not os.path.isdir(lm_output_path):
            os.makedirs(lm_output_path)

        # Extract the last file
        with open(lm_output_path + f_name, 'w') as fp:
            json.dump(dict_out, fp)

        return

    def execute_preprocessing(self):

        counter = 0

        if self.lang_model == 'CLDR_CLNER':
            # Define the tokenizer
            #medcharbert_tokenizer = BertTokenizer.from_pretrained('./character-bert-main/pretrained-models/bert-base-uncased/')
            # Load pre-trained CharacterBERT
            #medcharbert_model = CharacterBertModel.from_pretrained('./character-bert-main/pretrained-models/medical_character_bert/')
            
            # Load the trained weights (best validation loss)
            checkpoint_RE = torch.load(SAVED_MODEL_RE_PATH, map_location=torch.device('cpu'))
            tuned_characterBERT_RE = CharacterBertModel.from_pretrained(self.characterBERT_path + 'pretrained-models/medical_character_bert/')
            # Filter out unnecessary keys in order to use only the useful part of the trained model.
            trained_text_encoder_dict_RE = {k[14:]: v for k, v in checkpoint_RE['state_dict'].items() if k[14:] in tuned_characterBERT_RE.state_dict().keys()}
            tuned_characterBERT_RE.load_state_dict(trained_text_encoder_dict_RE)
            #tuned_characterBERT_RE = tuned_characterBERT_RE.to(self.device)


            # Define the model
            trained_model_NER = Model_NE(dropout_ne = 0,
                                         characterBERT_path = CHARACTER_BERT_PATH)
            # Load the trained weights (best validation loss)
            checkpoint_NER = torch.load(SAVED_MODEL_NER_PATH, map_location=torch.device('cpu'))
            trained_model_NER.load_state_dict(checkpoint_NER['state_dict'])
            #trained_model_NER = trained_model_NER.to(self.device)


        for i in range(len(self.original_dataset)):
            # Take the existing information from the generally pre-processed data set.
            tokens, ne_tags, relation_pairs, embeddings, ner_tags_numeric, filename = self.original_dataset[i]

            lang_model_name = self.lang_model + '_'
            # Take the embeddings using the selected language model
            
            # Add [CLS] and [SEP]
            word_pieces = ['[CLS]', *tokens, '[SEP]']
            
            # Convert token sequence into character indices
            indexer = CharacterIndexer()
            batch = [word_pieces]  # This is a batch with a single token sequence x
            batch_ids = indexer.as_padded_tensor(batch)

            # Feed batch to CharacterBERT & get the embeddings
            #embeddings_for_batch, _ = char_bert_model(batch_ids)
            #tmp_embeddings = embeddings_for_batch[0]
            #charbert_embeddings = tmp_embeddings[1:len(tmp_embeddings)-1]
                
                
            with torch.no_grad():
                extracted_embeddings_ready_RE_tmp = tuned_characterBERT_RE(batch_ids)
                out_text_encoder_NER_tmp = trained_model_NER(batch_ids)
                
			
            extracted_embeddings_ready_RE_tmp = extracted_embeddings_ready_RE_tmp[0]
            extracted_embeddings_ready_RE = extracted_embeddings_ready_RE_tmp[0][1:len(extracted_embeddings_ready_RE_tmp[0])-1]
            
            out_text_encoder_NER = out_text_encoder_NER_tmp[1:len(out_text_encoder_NER_tmp)-1]

            # Extracted the processed json file
            self.save_json(tokens,
                           ne_tags,
                           relation_pairs,
                           out_text_encoder_NER.tolist(),
                           extracted_embeddings_ready_RE.tolist(),
                           ner_tags_numeric,
                           filename,
                           lang_model_name)

            counter += 1
            if counter % 100 == 0:
                print('{} sentences processed.'.format(counter))

        return
        
        
cldr_clner_lm_process_data = CLDR_CLNER_Pre_Proc(processed_data_path = PROCESSED_DATA_PATH,
                                                 lang_model='CLDR_CLNER')
cldr_clner_lm_process_data.execute_preprocessing()