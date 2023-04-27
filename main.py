import torch
import json
#import argparse
import sys
import os
from classes.Data_Proc import Data_Proc
from torch.utils.data import Dataset, DataLoader
from classes.ADE_Dataset import ADE_Dataset
from classes.LM_Pre_Proc import LM_Pre_Proc
from dataloader_script import *
from helper_functions.get_chunks import *
from helper_functions.prepare_inputs import *
from helper_functions.create_input_target_tensors import *
#import helper_functions.biaffine_classifier as biaffine_classifier # Source: https://gist.github.com/JohnGiorgi/7472f3a523f53aed332ff2f8d6eff914
from IE_Model import *
from train_model import *
from helper_functions.plot_train import *
from evaluation_functions import *
from test_model import *
import random
import numpy as np
from common_config import *

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

    # Select the pre-trained language model
    #lang_model = 'CharBERT'
    #if lang_model == 'BERT' or lang_model == 'BioBERT' or lang_model == 'CharBERT' or lang_model == 'MedCharBERT':
    #    embedding_size = 768
    #elif lang_model == 'GloVe': # Didn't find any embeddings trained on medical text
    #    embedding_size = 100
    #elif lang_model == 'ELMo_default' or lang_model == 'ELMo_pubmed':
    #    embedding_size = 1024
    # CharBERT-Christos
    
    
    # Constants to control reprocessing options
    REPROCESS_DATA_SET = False
    RETRAIN_MODELS = True
    running_on_drive = False
    
    # Current directory
    #file_path = "C:/Users/plang/Desktop/2. Leuven/Thesis/1. Code"
    
    # Set directory paths
    #ADE_INITIAL_DATASET_PATH = file_path + "/dataset/ade_full_spert.json"
    #ADE_SPLITS_PATH = file_path + "/dataset/cv_splits.json"
    #PROCESSED_DATA_PATH = file_path + "/dataset/processed_data/"
    #SAVED_MODELS_PATH = file_path + "/saved_models/"
    #SAVED_CHECKPOINTS_PATH = file_path + "/saved_models/checkpoints/"
    
    # Set the hyperparameters
    #hyperparameters = {
    #    'batch_size': 8,
    #    'k_folds': 10,
    #    'epochs': 30,
    #    'early_stopping': 6,
    #    'early_stopping_threshold': 1.0, # to only consider significant changes
    #    'dropout_prob_ner': 0.1, # probability to randomly zero some of the elements of the input tensor
    #    'dropout_prob_rc': 0.1, # probability to randomly zero some of the elements of the input tensor
    #    'dataset_length': 4272, # Pre-computed
    #    'rc_loss_multiplier': 1,
    #    'learning_rate': 0.0003,
    #    'lr_annealing_factor': 0.25,
    #    'lr_annealing_patience': 2
    #}
    
    # Set the mapping of NER BIO tags to numeric values
    #mapping_ne_tags = {
    #    'B-DRUG': 1,
    #    'I-DRUG': 2,
    #    'B-AE': 3,
    #    'I-AE': 4,
    #    'O': 0
    #}
    
    # Define the running device by checking if a GPU is available
    #if torch.cuda.is_available():
    #    device = torch.device('cuda')
    #    print('GPU is available. Device set to', device)
    #else:
    #    device = torch.device('cpu')
    #    print('GPU is not available. Device set to', device)
    
    torch.backends.cudnn.enabled = False
    
    # Re-processing the raw data set    
    if REPROCESS_DATA_SET:
        process_data = Data_Proc(ADE_INITIAL_DATASET_PATH, PROCESSED_DATA_PATH)
        process_data.execute_preprocessing()
    
    if REPROCESS_DATA_SET:
        from transformers import AutoTokenizer, AutoModel, BertModel
        from transformers import logging
        logging.set_verbosity_error() # Suppress warnings
    
    if REPROCESS_DATA_SET:
        from allennlp.modules.elmo import Elmo, batch_to_ids
    
    # Creation of embeddings for the data set, using the selected Language Model 
    if REPROCESS_DATA_SET:
        lm_process_data = LM_Pre_Proc(PROCESSED_DATA_PATH, lang_model)
        lm_process_data.execute_preprocessing()
    
    
    if RETRAIN_MODELS == True:

        # always call this before training for deterministic results
        seed_everything()
        
        ### To save the losses per fold
        per_fold_train_loss_cmb = []
        per_fold_train_loss_ner = []
        per_fold_train_loss_rc = []
        per_fold_val_loss_cmb = []
        per_fold_val_loss_ner = []
        per_fold_val_loss_rc = []
        
        ### To average the results over the number of folds
        avg_ner_drug_prec = 0
        avg_ner_drug_rec = 0
        avg_ner_drug_f1 = 0
        avg_ner_ae_prec = 0
        avg_ner_ae_rec = 0
        avg_ner_ae_f1 = 0
        avg_rc_prec = 0
        avg_rc_rec = 0
        avg_rc_f1 = 0
        
        # 10-fold cross validation training
        for k_fold in range(hyperparameters['k_folds']):

            ### Define the loss functions
            rc_loss_function = nn.BCELoss()
            
            ### Initialize the model
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
            
            ### Prepare the dataloaders for each iteration of the 10-fold training
            dataloader_train, dataloader_valid, dataloader_test = prepare_dataloaders(split_num = k_fold+1)

            ### Train the model
            trained_model, \
            acc_training_loss_cmb, \
            acc_training_loss_ner, \
            acc_training_loss_rc, \
            acc_val_loss_cmb, \
            acc_val_loss_ner, \
            acc_val_loss_rc = train_model(k_fold, hyperparameters, ie_end2end_model,
                                          optimizer, scheduler, rc_loss_function, 
                                          dataloader_train, dataloader_valid)

            ### Store the losses of each fold
            per_fold_train_loss_cmb.append([acc_training_loss_cmb])
            per_fold_train_loss_ner.append([acc_training_loss_ner])
            per_fold_train_loss_rc.append([acc_training_loss_rc])
            per_fold_val_loss_cmb.append([acc_val_loss_cmb])
            per_fold_val_loss_ner.append([acc_val_loss_ner])
            per_fold_val_loss_rc.append([acc_val_loss_rc])
            
            print('For the {} fold we have the following values:\n'.format(str(k_fold+1)))
            print('The losses in the training subsets are ',acc_training_loss_cmb, '\n')
            print('The losses in the validation subsets are ',acc_val_loss_cmb, '\n')

            ### Plot the train and validation losses for the fold
            plot_train(k_fold, lang_model, hyperparameters, file_path,
                       acc_training_loss_cmb, acc_training_loss_ner,
                       acc_training_loss_rc, acc_val_loss_cmb,
                       acc_val_loss_ner, acc_val_loss_rc)
            
            ### Evaluate model on test set
            metrics_NER, metrics_REL = test_model(trained_model, dataloader_test)
            
            ### Save scores
            plot_path = file_path + '/plots/' + lang_model + '/'
            with open(plot_path + 'fold_' + str(k_fold+1) + "_metrics_NER.txt", "w") as fp:
                json.dump(metrics_NER, fp)  # encode dict into JSON
            with open(plot_path + 'fold_' + str(k_fold+1) + "_metrics_REL.txt", "w") as fp:
                json.dump(metrics_REL, fp)  # encode dict into JSON
                
            ### Accumulate the scores over the folds
            avg_ner_drug_prec += metrics_NER['DRUG']['Precision']
            avg_ner_drug_rec += metrics_NER['DRUG']['Recall']
            avg_ner_drug_f1 += metrics_NER['DRUG']['F1 score']
            avg_ner_ae_prec += metrics_NER['AE']['Precision']
            avg_ner_ae_rec += metrics_NER['AE']['Recall']
            avg_ner_ae_f1 += metrics_NER['AE']['F1 score']
            avg_rc_prec += metrics_REL['Precision']
            avg_rc_rec += metrics_REL['Recall']
            avg_rc_f1 += metrics_REL['F1 score']
            

        ### After k-fold is done, average the scores per task
        avg_ner_drug_prec /= hyperparameters['k_folds']
        avg_ner_drug_rec /= hyperparameters['k_folds']
        avg_ner_drug_f1 /= hyperparameters['k_folds']
        avg_ner_ae_prec /= hyperparameters['k_folds']
        avg_ner_ae_rec /= hyperparameters['k_folds']
        avg_ner_ae_f1 /= hyperparameters['k_folds']
        avg_rc_prec /= hyperparameters['k_folds']
        avg_rc_rec /= hyperparameters['k_folds']
        avg_rc_f1 /= hyperparameters['k_folds']

        k_fold_avg_scores = {
            'NER_DRUG': {
                'Recall': avg_ner_drug_rec,
                'Precision': avg_ner_drug_prec,
                'F1 score': avg_ner_drug_f1},
            'NER_AE': {
                 'Recall': avg_ner_ae_rec,
                 'Precision': avg_ner_ae_prec,
                 'F1 score': avg_ner_ae_f1},
            'RC': {
                 'Recall': avg_rc_rec,
                 'Precision': avg_rc_prec,
                 'F1 score': avg_rc_f1}
        }

        ### Save the hyperparameters' setup
        with open(plot_path + "hyperparameters.txt", "w") as fp:
            json.dump(hyperparameters, fp)  # encode dict into JSON

        ### Save k-fold average scores
        with open(plot_path + 'k_fold_avg_scores.txt', "w") as fp:
            json.dump(k_fold_avg_scores, fp)  # encode dict into JSON
        
        