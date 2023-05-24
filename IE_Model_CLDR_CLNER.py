import torch
from torch import nn
from torchcrf import CRF
import tqdm as notebook_tqdm
import helper_functions.biaffine_classifier as biaffine_classifier # Source: https://gist.github.com/JohnGiorgi/7472f3a523f53aed332ff2f8d6eff914
from common_config import *
from helper_functions.prepare_inputs import *

class IE_Model_CLDR_CLNER(nn.Module):
    def __init__(self, device, bilstm_input_size, bilstm_hidden_size, dropout_prob_ner, dropout_prob_rc):
        super(IE_Model_CLDR_CLNER, self).__init__()

        self.device = device

        ### ------------------ START OF NER TASK ------------------
        # Load the BiLSTM model
        self.ner_bilstm_model = nn.LSTM(input_size = bilstm_input_size,
                                        hidden_size = bilstm_hidden_size,
                                        num_layers = 2, # stacking two LSTMs together to form a stacked LSTM
                                        bias = True,
                                        bidirectional = True)

        # Dropout Layer
        self.dropout_layer_ner = torch.nn.Dropout(dropout_prob_ner)
        # Dense Layers aka Fully Connected Layers
        self.fully_connected_layer_1 = torch.nn.Linear(2*bilstm_hidden_size, 64, bias=True)
        self.fully_connected_layer_2 = torch.nn.Linear(64, 5, bias=True)

        # Initialize the CRF model with number of tags
        self.crf_model = CRF(num_tags = 5)
        self.crf_model.to(self.device)
        
        ### ------------------ END OF NER TASK -------------------
        #---------------------------------------------------------
        ### ------------------ START OF RC TASK ------------------
        # Activation function
        self.activation_function = nn.Sigmoid()

        # Load the BiLSTM model
        self.rc_bilstm_model = nn.LSTM(input_size=bilstm_input_size+5,
                                       hidden_size=bilstm_hidden_size,
                                       num_layers = 2,
                                       bias=True,
                                       bidirectional=True)

        # Dropout Layer
        self.dropout_layer_rc = torch.nn.Dropout(dropout_prob_rc)

        # The size of the output of the two FFNNs will be equal to the size of the
        # embeddings to avoid any compression of the information.
        tail_vector_representation_size = head_vector_representation_size = bilstm_input_size
        self.head_fc = torch.nn.Linear(2*bilstm_hidden_size, head_vector_representation_size, bias=False, device = self.device)  # Dense Layer aka Fully Connected Layer
        self.tail_fc = torch.nn.Linear(2*bilstm_hidden_size, tail_vector_representation_size, bias=False, device = self.device)  # Dense Layer aka Fully Connected Layer

        in_features = head_vector_representation_size
        out_features = 1 # Binary classification (classes are NEG & isEffectOf). Single 0/1 output
        self.biaffine_attention = biaffine_classifier.BiaffineAttention(in_features, out_features)

        ### ------------------ END OF RC TASK -------------------
        #---------------------------------------------------------

    def forward(self, word_embeddings_NER, word_embeddings_RE):
  
        ### ------------------ START OF NER TASK ------------------
        # Pass the emdeddings through the BiLSTM
        word_embeddings_NER = word_embeddings_NER.to(self.device)
        word_embeddings_RE = word_embeddings_RE.to(self.device)
        
        bilstm_output = self.ner_bilstm_model(word_embeddings_NER)

        # Pass the BiLSTM output from the Dropout layer
        tmp_output = self.dropout_layer_ner(bilstm_output[0])   

        # Pass the output through the dense layers to drop to 5 logits (one for each BIO tag)
        ner_model_output = self.fully_connected_layer_1(tmp_output)
        ner_model_output = self.fully_connected_layer_2(ner_model_output) # To enable Dropout, change bilstm_output[0] ==> tmp_output
        ner_model_output = ner_model_output.unsqueeze_(0)
        ner_model_output = ner_model_output.permute(1,0,2)
        
        ### ------------------ END OF NER TASK -------------------
        #---------------------------------------------------------
        
        ### ------------------ CRF PREDICTIONS -------------------
        # Predict the most likely tag sequence for the given output of the NER BiLSTM using the Viterbi algorithm
        crf_preds = self.crf_model.decode(ner_model_output)  #==> [[3, 1, 3], [0, 1, 0]]
        predicted_NER_tags = torch.tensor(crf_preds[0], dtype=torch.long)
        predicted_NER_tags = predicted_NER_tags.to(self.device)
        ### ------------------ END OF NER LOSS AND PREDICTIONS -------------------
        #---------------------------------------------------------
        
        ### ------------------ START OF RC TASK ------------------
        # Prepare the inputs for the RC model
        tag_n_token_embeddings, \
        drug_entity_indexes, \
        effect_entity_indexes = prepare_inputs(token_ner_tags = predicted_NER_tags,
                                               token_embeddings = word_embeddings_RE)
        
        # Pass the concatenated vector (NER tag embedding + word piece embeddings) through the RC BiLSTM
        rc_bilstm_model_output = self.rc_bilstm_model(tag_n_token_embeddings)

        # Pass the BiLSTM output from the Dropout layer
        model_dropped_output = self.dropout_layer_rc(rc_bilstm_model_output[0])

        ### Pass the drug entities through the head dense layers
        all_potential_pairs = []
        all_potential_pairs_biaf_out = torch.tensor([], device=self.device)
        all_potential_pairs_probabilities = torch.tensor([], device=self.device)

        for i in range(len(drug_entity_indexes)):
            h_head_i = self.head_fc( model_dropped_output[drug_entity_indexes[i]] )
            h_head_i = h_head_i.unsqueeze_(0)

            for j in range(len(effect_entity_indexes)):
                h_tail_i = self.tail_fc( model_dropped_output[effect_entity_indexes[j]] )
                # Pass the (head,tail) relation pair through the biaffine classifier
                h_tail_i = h_tail_i.unsqueeze_(0)
                biaffine_output = self.biaffine_attention(h_head_i, h_tail_i)

                # Use the activation function on the output of the biaffine classifier
                # The Sigmoid function maps the biaffine output to a probability which will be used for BCE loss later
                predicted_relation_prob = self.activation_function(biaffine_output[0])

                # Store indexes of the predicted pairs
                all_potential_pairs.append([drug_entity_indexes[i], effect_entity_indexes[j]])
                all_potential_pairs_probabilities = torch.cat((all_potential_pairs_probabilities,predicted_relation_prob), dim=-1)

                
        ### ------------------ END OF RC TASK -------------------
        #---------------------------------------------------------      

        return ner_model_output, all_potential_pairs, all_potential_pairs_probabilities