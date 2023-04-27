import torch

# Import module from parent directory
import sys
sys.path.append('C:/Users/plang/Desktop/2. Leuven/Thesis/1. Code')
from common_config import *

def create_input_target_tensors(device, predicted_pairs, predicted_pairs_probabilities, actual_pairs):
    # This functions gets the relation pairs that the model predicted
    # and the actual relation pairs, and returns the corresponding input
    # and target tensors for the calculation of the loss.

    if len(predicted_pairs) != 0:
        # If the model predicted at least one relation pair

        # Create input & target tensors for relation classes
        input_probab = torch.tensor([], device=device, dtype=torch.float32, requires_grad=True)
        input_probab = torch.cat((input_probab, predicted_pairs_probabilities), dim=0)

        target = torch.tensor([], device=device, dtype=torch.float32)

        # Iterate through the predicted relation pairs and create the target tensors
        for i in range(len(predicted_pairs)):
            if predicted_pairs[i] in actual_pairs:
                target = torch.cat((target, torch.ones(1,device=device)), dim=0)
            else:
                target = torch.cat((target, torch.zeros(1,device=device)), dim=0)

        # For the unidentified gold pairs
        for i in range(len(actual_pairs)):
            if actual_pairs[i] not in predicted_pairs:
                input_probab = torch.cat((input_probab, torch.zeros(1,device=device)), dim=0)
                target = torch.cat((target, torch.ones(1,device=device)), dim=0)

    else:
        # If the model did not identify any relation pairs
        # Create criterion input & target tensors for the unidentified gold pairs
        input_probab = torch.zeros(len(actual_pairs),device=device, requires_grad=True)
        target = torch.ones(len(actual_pairs),device=device)

    return input_probab, target