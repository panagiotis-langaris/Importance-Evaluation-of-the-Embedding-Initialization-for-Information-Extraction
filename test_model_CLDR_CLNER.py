import torch
from main import *
from helper_functions.get_chunks import *
from evaluation_functions import *

def test_model_CLDR_CLNER(model, device, mapping_ne_tags, dataloader_test):

    # Call model. eval() to set dropout and batch normalization layers to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    model.eval()

    # Initialize confusion matrix scores
    tpsClassesNER = {'DRUG': 0,
                     'AE': 0}
    fpsClassesNER = {'DRUG': 0,
                     'AE': 0}
    fnsClassesNER = {'DRUG': 0,
                     'AE': 0}

    tp_rel_global = fp_rel_global = fn_rel_global = 0

    for batch in dataloader_test:

        # Take the inputs from the batch  
        tokens, ne_tags, relation_pairs, embeddings_NER, embeddings_RE, ner_tags_numeric, filename = batch

        # Run the forward pass for the batch
        for in1, in2, in3, in4, in5, in6, in7 in zip(tokens,
                                                     ne_tags,
                                                     relation_pairs,
                                                     embeddings_NER,
                                                     embeddings_RE,
                                                     ner_tags_numeric,
                                                     filename):

            # Place the inputs in the selected device (GPU or CPU)
            in4 = torch.tensor(in4, dtype=torch.float32)
            in4 = in4.to(device)
            in5 = torch.tensor(in5, dtype=torch.float32)
            in5 = in5.to(device)

            # Pass the embeddings through the Information Extraction Model
            ner_model_output, \
            all_potential_pairs, \
            all_potential_pairs_probabilities = model(word_embeddings_NER = in4,
                                                      word_embeddings_RE = in4) # Changed to in4 instead of in5 to test only CLNER embeddings
                        
            ### --------------  NER Task Predictions --------------
            # Predict the most likely tag sequence for the given output of the NER BiLSTM using the Viterbi algorithm
            crf_preds = model.crf_model.decode(ner_model_output)  #==> [[3, 1, 3], [0, 1, 0]]

            predicted_NER_tags = torch.tensor(crf_preds[0], dtype=torch.long)
            predicted_NER_tags = predicted_NER_tags.to(device)

            # Evaluate the NER tags' predictions
            gold_seq = in6
            gold_chunks = get_chunks(gold_seq, mapping_ne_tags)

            predicted_seq = predicted_NER_tags.tolist()
            predicted_chunks = get_chunks(predicted_seq, mapping_ne_tags)

            for lab_idx in range(len(predicted_chunks)):
                if predicted_chunks[lab_idx] in gold_chunks:
                    tpsClassesNER[predicted_chunks[lab_idx][0]] += 1
                else:
                    fpsClassesNER[predicted_chunks[lab_idx][0]] += 1

            for lab_idx in range(len(gold_chunks)):
                if gold_chunks[lab_idx] not in predicted_chunks:
                    fnsClassesNER[gold_chunks[lab_idx][0]] += 1

            ### --------------  RC Task Predictions --------------
            predicted_pairs = []
            for i in range(len(all_potential_pairs)):
                if all_potential_pairs_probabilities[i] >= 0.5:
                    predicted_pairs.append(all_potential_pairs[i])          
            
            # Evaluate relation extraction (predicted_rel, gold_rel, gold_chunks, predicted_chunks)
            tp_rel, fp_rel, fn_rel = evaluate_RE(predicted_rel = predicted_pairs,
                                                 gold_rel = in3,
                                                 gold_chunks = gold_chunks,
                                                 predicted_chunks = predicted_chunks)
            
            tp_rel_global += tp_rel
            fp_rel_global += fp_rel
            fn_rel_global += fn_rel
    
    metrics_NER = get_metrics_NER(tpsClassesNER, fpsClassesNER, fnsClassesNER)
    
    if tp_rel_global == 0:
        print('NOT WORKING------------------------------------------!!!!!!!!!!!!!!')
        metrics_REL = {'Recall': -1,
            'Precision': -1,
            'F1 score': -1}
    else:
        metrics_REL = get_metrics_rel(tp_rel_global, fp_rel_global, fn_rel_global)
    
    return metrics_NER, metrics_REL