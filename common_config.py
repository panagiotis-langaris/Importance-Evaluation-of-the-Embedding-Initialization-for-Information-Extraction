import torch

# Select the pre-trained language model
lang_model = 'ELMo_pubmed'
if lang_model == 'BERT' or lang_model == 'BioBERT' or lang_model == 'CharBERT' or lang_model == 'MedCharBERT':
    embedding_size = 768
elif lang_model == 'GloVe': # Didn't find any embeddings trained on medical text
    embedding_size = 100
elif lang_model == 'ELMo_default' or lang_model == 'ELMo_pubmed':
    embedding_size = 1024
# CharBERT-Christos


# Current directory
file_path = "C:/Users/plang/Desktop/2. Leuven/Thesis/1. Code"

# Set directory paths
ADE_INITIAL_DATASET_PATH = file_path + "/dataset/ade_full_spert.json"
ADE_SPLITS_PATH = file_path + "/dataset/cv_splits.json"
PROCESSED_DATA_PATH = file_path + "/dataset/processed_data/"
SAVED_MODELS_PATH = file_path + "/saved_models/"
SAVED_CHECKPOINTS_PATH = file_path + "/saved_models/checkpoints/"

# Set the hyperparameters
hyperparameters = {
    'batch_size': 8,
    'k_folds': 10,
    'epochs': 30,
    'early_stopping': 6,
    'early_stopping_threshold': 1.0, # to only consider significant changes
    'dropout_prob_ner': 0.1, # probability to randomly zero some of the elements of the input tensor
    'dropout_prob_rc': 0.1, # probability to randomly zero some of the elements of the input tensor
    'dataset_length': 4272, # Pre-computed
    'rc_loss_multiplier': 1,
    'learning_rate': 0.0003,
    'lr_annealing_factor': 0.25,
    'lr_annealing_patience': 2
}

# Set the mapping of NER BIO tags to numeric values
mapping_ne_tags = {
    'B-DRUG': 1,
    'I-DRUG': 2,
    'B-AE': 3,
    'I-AE': 4,
    'O': 0
}

# Define the running device by checking if a GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU is available. Device set to', device)
else:
    device = torch.device('cpu')
    print('GPU is not available. Device set to', device)