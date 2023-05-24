# Current directory
file_path = "C:/Users/plang/Desktop/2. Leuven/Thesis/1. Code"

# Set directory paths
ADE_INITIAL_DATASET_PATH = file_path + "/dataset/ade_full_spert.json"
ADE_SPLITS_PATH = file_path + "/dataset/cv_splits.json"
PROCESSED_DATA_PATH = file_path + "/dataset/processed_data/"
SAVED_MODELS_PATH = file_path + "/saved_models/"

# Set the hyperparameters
hyperparameters = {
    'batch_size': 8,
    'epochs': 30,
    'early_stopping': 7,
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
#mapping_ne_tags = {
#    'B-DRUG': 1,
#    'I-DRUG': 2,
#    'B-AE': 3,
#    'I-AE': 4,
#    'O': 0
#}