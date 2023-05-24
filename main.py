import torch
import json
import sys
from sys import exit
import os
from common_config import *
from classes.Data_Proc import Data_Proc
from torch.utils.data import Dataset, DataLoader
from classes.ADE_Dataset import ADE_Dataset, ADE_Dataset_CLDR_CLNER
from classes.LM_Pre_Proc import LM_Pre_Proc
from dataloader_script import *
from helper_functions.get_chunks import *
from helper_functions.prepare_inputs import *
from helper_functions.create_input_target_tensors import *
from IE_Model import *
from IE_Model_CLDR_CLNER import *
from train_model import *
from train_model_CLDR_CLNER import *
from helper_functions.plot_train import *
from evaluation_functions import *
from test_model import *
from test_model_CLDR_CLNER import *
import random
import numpy as np

if len(sys.argv) > 1:
    lang_model = str(sys.argv[1])
    k_fold_num = int(sys.argv[2])
else:
    print('Error. Missing parameter of k-fold number.')
    print('Program terminated.')
    exit()

# GloVe libraries
#import torchtext # This was only for running_on_drive
#from torchtext.vocab import GloVe # This was only for running_on_drive

# BERT Libraries
from transformers import AutoTokenizer, AutoModel, BertModel
from transformers import logging
logging.set_verbosity_error() # Suppress warnings

# CharacterBERT libraries
#%cd C:\Users\plang\anaconda3\envs\character-bert # Change path to run character-bert
#from transformers import BertTokenizer
#from modeling.character_bert import CharacterBertModel
#from utils.character_cnn import CharacterIndexer

# ELMo libraries
from allennlp.modules.elmo import Elmo, batch_to_ids

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    # Define the running device by checking if a GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU is available. Device set to', device)
    else:
        device = torch.device('cpu')
        print('GPU is not available. Device set to', device)

    # Set the embedding_size for the selected pre-trained language model
    if lang_model == 'BERT' or lang_model == 'BioBERT' or lang_model == 'CharBERT' or lang_model == 'MedCharBERT' or lang_model == 'CLDR_CLNER':
        embedding_size = 768
    elif lang_model == 'GloVe': # Didn't find any embeddings trained on medical text
        embedding_size = 100
    elif lang_model == 'ELMo_default' or lang_model == 'ELMo_pubmed':
        embedding_size = 1024
    
    # Set the mapping of NER BIO tags to numeric values
    mapping_ne_tags = {
        'B-DRUG': 1,
        'I-DRUG': 2,
        'B-AE': 3,
        'I-AE': 4,
        'O': 0
    }
    
    ### Constants to control reprocessing options
    REPROCESS_DATA_SET = False
    RETRAIN_MODELS = True
    running_on_drive = False
    
    torch.backends.cudnn.enabled = False
    
    ### Re-processing the raw data set    
    if REPROCESS_DATA_SET:
        process_data = Data_Proc(ADE_INITIAL_DATASET_PATH, PROCESSED_DATA_PATH)
        process_data.execute_preprocessing()
    
    if REPROCESS_DATA_SET:
        from transformers import AutoTokenizer, AutoModel, BertModel
        from transformers import logging
        logging.set_verbosity_error() # Suppress warnings
    
    if REPROCESS_DATA_SET:
        from allennlp.modules.elmo import Elmo, batch_to_ids
    
    ### Creation of embeddings for the data set, using the selected Language Model 
    if REPROCESS_DATA_SET:
        lm_process_data = LM_Pre_Proc(PROCESSED_DATA_PATH, lang_model)
        lm_process_data.execute_preprocessing()
    
    ### Always call this before training for deterministic results
    seed_everything()
    
    ### Prepare the dataloaders for each iteration of the 10-fold training
    dataloader_train, dataloader_valid, dataloader_test = prepare_dataloaders(lang_model = lang_model,
                                                                              split_num = k_fold_num+1)
    
    if RETRAIN_MODELS == True:
    
        ### To save the losses per fold
        per_fold_train_loss_cmb = []
        per_fold_train_loss_ner = []
        per_fold_train_loss_rc = []
        per_fold_val_loss_cmb = []
        per_fold_val_loss_ner = []
        per_fold_val_loss_rc = []

        ### Define the loss functions
        rc_loss_function = nn.BCELoss()
        
        ### Initialize the model
        if lang_model == 'CLDR_CLNER':
            ie_end2end_model = IE_Model_CLDR_CLNER(device = device,
                                                   bilstm_input_size = embedding_size,
                                                   bilstm_hidden_size = embedding_size,
                                                   dropout_prob_ner = hyperparameters['dropout_prob_ner'],
                                                   dropout_prob_rc = hyperparameters['dropout_prob_rc'])
        else:
            ie_end2end_model = IE_Model(device = device,
                                        bilstm_input_size = embedding_size,
                                        bilstm_hidden_size = embedding_size,
                                        dropout_prob_ner = hyperparameters['dropout_prob_ner'],
                                        dropout_prob_rc = hyperparameters['dropout_prob_rc'])
        ie_end2end_model.to(device)
        
        ### Initialize the optimizer
        optimizer = optim.Adam(ie_end2end_model.parameters(),
                               lr = hyperparameters['learning_rate'],
                               weight_decay = 0.000001)

        ### Learning Rate Annealing
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor = hyperparameters['lr_annealing_factor'],
                                                         patience = hyperparameters['lr_annealing_patience'],
                                                         #threshold = 0.0001, # best * ( 1 - threshold ) # Threshold for measuring the new optimum, to only focus on significant changes.
                                                         verbose = True)
        
        ### Train the model
        if lang_model == 'CLDR_CLNER':
            trained_model, \
            acc_training_loss_cmb, \
            acc_training_loss_ner, \
            acc_training_loss_rc, \
            acc_val_loss_cmb, \
            acc_val_loss_ner, \
            acc_val_loss_rc = train_model_CLDR_CLNER(device, lang_model, k_fold_num, hyperparameters,
                                                     ie_end2end_model, optimizer, scheduler, rc_loss_function,
                                                     dataloader_train, dataloader_valid)
        else:
            trained_model, \
            acc_training_loss_cmb, \
            acc_training_loss_ner, \
            acc_training_loss_rc, \
            acc_val_loss_cmb, \
            acc_val_loss_ner, \
            acc_val_loss_rc = train_model(device, lang_model, k_fold_num, hyperparameters,
                                          ie_end2end_model, optimizer, scheduler, rc_loss_function,
                                          dataloader_train, dataloader_valid)

        ### Store the losses of each fold
        per_fold_train_loss_cmb.append([acc_training_loss_cmb])
        per_fold_train_loss_ner.append([acc_training_loss_ner])
        per_fold_train_loss_rc.append([acc_training_loss_rc])
        per_fold_val_loss_cmb.append([acc_val_loss_cmb])
        per_fold_val_loss_ner.append([acc_val_loss_ner])
        per_fold_val_loss_rc.append([acc_val_loss_rc])
        
        print('For the {} fold we have the following values:\n'.format(str(k_fold_num+1)))
        print('The losses in the training subsets are ',acc_training_loss_cmb, '\n')
        print('The losses in the validation subsets are ',acc_val_loss_cmb, '\n')

        ### Plot the train and validation losses for the fold
        plot_train(k_fold_num, lang_model, hyperparameters, file_path,
                   acc_training_loss_cmb, acc_training_loss_ner,
                   acc_training_loss_rc, acc_val_loss_cmb,
                   acc_val_loss_ner, acc_val_loss_rc)
        
    ### Initialize the best model
    if lang_model == 'CLDR_CLNER':
        best_model = IE_Model_CLDR_CLNER(device = device,
                                         bilstm_input_size = embedding_size,
                                         bilstm_hidden_size = embedding_size,
                                         dropout_prob_ner = hyperparameters['dropout_prob_ner'],
                                         dropout_prob_rc = hyperparameters['dropout_prob_rc'])                          
    else:
        best_model = IE_Model(device = device,
                              bilstm_input_size = embedding_size,
                              bilstm_hidden_size = embedding_size,
                              dropout_prob_ner = hyperparameters['dropout_prob_ner'],
                              dropout_prob_rc = hyperparameters['dropout_prob_rc'])    
    
    best_model.to(device)
    # Load best performing model weights
    best_model.load_state_dict(torch.load(SAVED_MODELS_PATH + lang_model + '_fold_' + str(k_fold_num) + '_best_model.pth'))
    
    ### Evaluate model on test set
    if lang_model == 'CLDR_CLNER':
        metrics_NER, metrics_REL = test_model_CLDR_CLNER(best_model, device, mapping_ne_tags, dataloader_test)
    else:
        metrics_NER, metrics_REL = test_model(best_model, device, mapping_ne_tags, dataloader_test)

    ### Save scores
    plot_path = file_path + '/plots/' + lang_model + '/'
    with open(plot_path + 'fold_' + str(k_fold_num+1) + "_metrics_NER.txt", "w") as fp:
        json.dump(metrics_NER, fp)  # encode dict into JSON
    with open(plot_path + 'fold_' + str(k_fold_num+1) + "_metrics_REL.txt", "w") as fp:
        json.dump(metrics_REL, fp)  # encode dict into JSON
    
    ### Save the hyperparameters' setup
    with open(plot_path + "hyperparameters.txt", "w") as fp:
        json.dump(hyperparameters, fp)  # encode dict into JSON