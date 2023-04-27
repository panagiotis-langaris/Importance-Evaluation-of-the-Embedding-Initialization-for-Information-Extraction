import torch
import torch.optim as optim
from random import sample
import gc
from common_config import *
from helper_functions.create_input_target_tensors import *

def train_model(k_fold, hyperparameters, model, optimizer, scheduler, rc_loss_probab, dataloader_train, dataloader_valid):

    acc_training_loss_cmb = []
    acc_training_loss_ner = []
    acc_training_loss_rc = []
    
    acc_val_loss_cmb = []
    acc_val_loss_ner = []
    acc_val_loss_rc = []
    
    # Initialize the minimum validation loss with a large value
    valid_loss_min = 1000000000

    # For applying early stopping
    valid_no_improvement = 0
    
    for epoch in range(hyperparameters['epochs']):

        print('__________________________________________________')
        print('---------   Fold k = {}  /// Epoch: {}   ---------'.format(k_fold+1, epoch + 1))
        print('__________________________________________________')

        ####################################################################
        ##############          Training step             ##################
        ####################################################################
        print('-------------------')
        print('Training Step')
        print('-------------------')

        model.train()
        
        batch_counter_1 = 0
        
        training_loss_cmb = 0.0
        training_loss_ner = 0.0
        training_loss_rc = 0.0

        for batch in dataloader_train:

            # Take the inputs from the batch  
            tokens, ne_tags, relation_pairs, embeddings, ner_tags_numeric, filename = batch

            # Initialize batch's NER loss
            ner_loss_batch = 0.0
            # Initialize criterion input and target lists at GPU
            rc_criterion_input_batch_probab = torch.tensor([], dtype=torch.float32, device='cuda', requires_grad=True)
            rc_criterion_target_batch = torch.tensor([], dtype=torch.float32, device='cuda')

            # Run the forward pass for the batch
            for in1, in2, in3, in4, in5, in6 in zip(tokens,
                                                    ne_tags,
                                                    relation_pairs,
                                                    embeddings,
                                                    ner_tags_numeric,
                                                    filename):

                # Place the inputs in the selected device (GPU or CPU)
                in4 = torch.tensor(in4, dtype=torch.float32)
                in4 = in4.to(device)
                in5 = torch.tensor(in5, dtype=torch.long)
                in5 = in5.to(device)

                # Pass the embeddings through the Information Extraction Model
                ner_model_output, \
                all_potential_pairs, \
                all_potential_pairs_probabilities = model(word_embeddings = in4)
                
                # Prepare gold BIO tags for CRF model 
                crf_gold_tags = in5.unsqueeze_(0)
                crf_gold_tags = crf_gold_tags.permute(1,0)
                # Calculate NER loss using the CRF model
                log_likelihood = model.crf_model(ner_model_output, crf_gold_tags)
                # The NER task loss of the batch is the summary of the individual negative log likelihoods
                ner_loss_batch += -log_likelihood
                
                # Create criterion input and target labels
                rc_input_probab, \
                rc_target = create_input_target_tensors(device = device,
                                                        predicted_pairs = all_potential_pairs,
                                                        predicted_pairs_probabilities = all_potential_pairs_probabilities,
                                                        actual_pairs = in3)

                # Append criterion inputs and targets in the batch level
                rc_criterion_input_batch_probab = torch.cat((rc_criterion_input_batch_probab, rc_input_probab), 0)
                rc_criterion_target_batch = torch.cat((rc_criterion_target_batch, rc_target), 0)

            # Calculate RC loss of the batch
            rc_loss_batch_probab = rc_loss_probab(rc_criterion_input_batch_probab, rc_criterion_target_batch.float())

            # Add the two losses
            loss_batch = ner_loss_batch + hyperparameters['rc_loss_multiplier']*rc_loss_batch_probab

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward() is called.
            optimizer.zero_grad()

            # Run a backpropagation pass
            loss_batch.backward()

            # Gradient descent step
            optimizer.step()

            # Add the loss of the batch
            training_loss_cmb += loss_batch.data.item()
            training_loss_ner += ner_loss_batch.data.item()
            training_loss_rc += rc_loss_batch_probab.data.item()

            # Increment the counter (number of batches)
            batch_counter_1 += 1

            if batch_counter_1 % 50 == 0:
                print('{} batches completed.'.format(batch_counter_1))  

        # Find the average training loss over the batches
        training_loss_cmb /= batch_counter_1
        training_loss_ner /= batch_counter_1
        training_loss_rc /= batch_counter_1

        # Save the training loss
        acc_training_loss_cmb.append(training_loss_cmb)
        acc_training_loss_ner.append(training_loss_ner)
        acc_training_loss_rc.append(training_loss_rc)

        # Print the loss at the end of each epoch
        print('Epoch\'s Training Loss')
        print('Combined\tNER\tRC')
        print(' {:.2f}\t\t{:.2f}\t{:.2f}'.format(training_loss_cmb, training_loss_ner, training_loss_rc))
        print('_________________________')

        ####################################################################
        #############          Validation step            ##################
        ####################################################################
        print('---------------------')
        print('Validation Step')
        print('---------------------')

        model.eval()
        
        batch_counter_2 = 0
        
        valid_loss_cmb = 0.0
        valid_loss_ner = 0.0
        valid_loss_rc = 0.0

        with torch.no_grad():
            for batch in dataloader_valid:
                # Take the inputs from the batch  
                tokens, ne_tags, relation_pairs, embeddings, ner_tags_numeric, filename = batch

                # Initialize batch's NER loss
                val_ner_loss_batch = 0.0 # batch_negative_log_likelihood_val = 0.0

                # Initialize criterion input and target lists at GPU
                rc_criterion_input_batch_probab = torch.tensor([], dtype=torch.float32, device='cuda')
                rc_criterion_target_batch = torch.tensor([], dtype=torch.float32, device='cuda')

                # Run the forward pass for the batch
                for in1, in2, in3, in4, in5, in6 in zip(tokens,
                                                        ne_tags,
                                                        relation_pairs,
                                                        embeddings,
                                                        ner_tags_numeric,
                                                        filename):

                    # Place the inputs in the selected device (GPU or CPU)
                    in4 = torch.tensor(in4, dtype=torch.float32)
                    in4 = in4.to(device)
                    in5 = torch.tensor(in5, dtype=torch.long)
                    in5 = in5.to(device)

                    
                    # Pass the embeddings through the Information Extraction Model
                    ner_model_output, \
                    all_potential_pairs, \
                    all_potential_pairs_probabilities = model(word_embeddings = in4)
                    
                    # Prepare gold BIO tags for CRF model
                    crf_gold_tags = in5.unsqueeze_(0)
                    crf_gold_tags = crf_gold_tags.permute(1,0)
                    # Calculate NER loss using the CRF model
                    log_likelihood = model.crf_model(ner_model_output, crf_gold_tags)
                    # The NER task loss of the batch is the summary of the individual negative log likelihoods
                    val_ner_loss_batch += -log_likelihood

                    # Create criterion input and target labels
                    rc_input_probab, \
                    rc_target = create_input_target_tensors(device = device, 
                                                            predicted_pairs = all_potential_pairs,
                                                            predicted_pairs_probabilities = all_potential_pairs_probabilities,
                                                            actual_pairs = in3)

                    # Append criterion inputs and targets in the batch level
                    rc_criterion_input_batch_probab = torch.cat((rc_criterion_input_batch_probab, rc_input_probab), 0)
                    rc_criterion_target_batch = torch.cat((rc_criterion_target_batch, rc_target), 0)

                # Calculate RC loss of the batch
                rc_loss_batch_val_probab = rc_loss_probab(rc_criterion_input_batch_probab, rc_criterion_target_batch.float())

                # Add the two losses
                loss_batch_val_cmb = val_ner_loss_batch + hyperparameters['rc_loss_multiplier']*rc_loss_batch_val_probab

                # Add the loss of the batch
                valid_loss_cmb += loss_batch_val_cmb.data.item()
                valid_loss_ner += val_ner_loss_batch.data.item()
                valid_loss_rc += rc_loss_batch_val_probab.data.item()

                # Increment the counter (number of batches)
                batch_counter_2 += 1

                if batch_counter_2 % 10 == 0:
                    print('{} batches completed.'.format(batch_counter_2))

        # Find the average validation loss over the batches
        valid_loss_cmb /= batch_counter_2
        valid_loss_ner /= batch_counter_2
        valid_loss_rc /= batch_counter_2

        # Save the validation loss
        acc_val_loss_cmb.append(valid_loss_cmb)
        acc_val_loss_ner.append(valid_loss_ner)
        acc_val_loss_rc.append(valid_loss_rc)

        # Print the losses at the end of each epoch
        print('Epoch\'s Avg Training Loss')
        print('Combined\tNER\t\tRC')
        print(' {:.2f}\t\t{:.2f}\t\t{:.2f}'.format(training_loss_cmb, training_loss_ner, training_loss_rc))
        print('- - - - - - - - - - - - -')
        print('Epoch\'s Avg Validation Loss')
        print('Combined\tNER\t\tRC')
        print(' {:.2f}\t\t{:.2f}\t\t{:.2f}'.format(valid_loss_cmb, valid_loss_ner, valid_loss_rc))
        print('_________________________')

        # Check if the validation loss has been reduced in order to update the "best" checkpoint
        if valid_loss_cmb<= valid_loss_min:
            # Update the early stopping counter
            valid_no_improvement = 0
            print('Validation loss decreased from ({:.4f} --> {:.4f})!'.format(valid_loss_min, valid_loss_cmb))
            print('##########################')
            # Update the minimum validation loss
            valid_loss_min = valid_loss_cmb
        else:
            # No validation improvement
            valid_no_improvement += 1
        
        # Check for early stopping
        if valid_no_improvement >= hyperparameters['early_stopping']:
            print('Early stopping after {} epochs.'.format(str(epoch + 1)))
            break
        
        # Learning Rate Annealing
        # At the end of each epoch, adjust the learning rate based on combined validation loss
        new_lr = optimizer.param_groups[0]['lr']
        print("Epoch:", epoch+1, "Starting learning rate:", new_lr)
        print("Scheduler state:", scheduler.state_dict())
        print('---------------------------------------------------------------------')
        
        scheduler.step(valid_loss_cmb)
        
        new_lr = optimizer.param_groups[0]['lr']
        print("Epoch:", epoch+1, "New learning rate:", new_lr)
        print("Scheduler state:", scheduler.state_dict())
        print('---------------------------------------------------------------------')
        
        torch.cuda.empty_cache()
        gc.collect()
        
    return model, acc_training_loss_cmb, acc_training_loss_ner, acc_training_loss_rc, acc_val_loss_cmb, acc_val_loss_ner, acc_val_loss_rc