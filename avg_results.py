'''
    This function calculates the average evaluation scores for the k-folds.
    It also counts the problematic / non-existing folds.
'''

import json
import os
import sys
from sys import exit

if len(sys.argv) > 1:
    lang_model = str(sys.argv[1])
else:
    print('Error. Missing parameter of k-fold number.')
    print('Program terminated.')
    exit()
    

file_dir = 'C:/Users/plang/Desktop/2. Leuven/Thesis/1. Code/plots/' + lang_model

num_files_NER = 0
total_recall_NER_DRUG = 0
total_precision_NER_DRUG = 0
total_f1_NER_DRUG = 0
files_with_minus_one_NER_DRUG = 0
total_recall_NER_AE = 0
total_precision_NER_AE = 0
total_f1_NER_AE = 0
files_with_minus_one_NER_AE = 0

num_files_RE = 0
total_recall_RE = 0
total_precision_RE = 0
total_f1_RE = 0
files_with_minus_one_RE = 0

for filename in os.listdir(file_dir):
    if "_metrics_NER" in filename:
        with open(os.path.join(file_dir, filename), "r") as file_NER:
            content_NER = file_NER.read()
            try:
                scores_NER = json.loads(content_NER)
                
                recall_NER_DRUG = scores_NER['DRUG'].get("Recall")
                precision_NER_DRUG = scores_NER['DRUG'].get("Precision")
                f1_NER_DRUG = scores_NER['DRUG'].get("F1 score")

                recall_NER_AE = scores_NER['AE'].get("Recall")
                precision_NER_AE = scores_NER['AE'].get("Precision")
                f1_NER_AE = scores_NER['AE'].get("F1 score")
                
                if recall_NER_DRUG != -1 and precision_NER_DRUG != -1 and f1_NER_DRUG != -1:
                    total_recall_NER_DRUG += recall_NER_DRUG
                    total_precision_NER_DRUG += precision_NER_DRUG
                    total_f1_NER_DRUG += f1_NER_DRUG
                else:
                    files_with_minus_one_NER_DRUG += 1
                    
                if recall_NER_AE != -1 and precision_NER_AE != -1 and f1_NER_AE != -1:
                    total_recall_NER_AE += recall_NER_AE
                    total_precision_NER_AE += precision_NER_AE
                    total_f1_NER_AE += f1_NER_AE
                else:
                    files_with_minus_one_NER_AE += 1
                    
                num_files_NER += 1
            except json.JSONDecodeError:
                print(f"Error parsing JSON in file: {filename}")

    if "_metrics_REL" in filename:
        with open(os.path.join(file_dir, filename), "r") as file_RE:
            content_RE = file_RE.read()
            try:
                scores_RE = json.loads(content_RE)
                recall_RE = scores_RE.get("Recall")
                precision_RE = scores_RE.get("Precision")
                f1_RE = scores_RE.get("F1 score")
                if recall_RE != -1 and precision_RE != -1 and f1_RE != -1:
                    total_recall_RE += recall_RE
                    total_precision_RE += precision_RE
                    total_f1_RE += f1_RE
                else:
                    files_with_minus_one_RE += 1
                num_files_RE += 1
            except json.JSONDecodeError:
                print(f"Error parsing JSON in file: {filename}")

print('Number of files for NER is', num_files_NER)

average_recall_NER_DRUG = total_recall_NER_DRUG / (num_files_NER - files_with_minus_one_NER_DRUG) if num_files_NER != 0 else 0
average_precision_NER_DRUG = total_precision_NER_DRUG / (num_files_NER - files_with_minus_one_NER_DRUG) if num_files_NER != 0 else 0
average_f1_NER_DRUG = total_f1_NER_DRUG / (num_files_NER - files_with_minus_one_NER_DRUG) if num_files_NER != 0 else 0

average_recall_NER_AE = total_recall_NER_AE / (num_files_NER - files_with_minus_one_NER_AE) if num_files_NER != 0 else 0
average_precision_NER_AE = total_precision_NER_AE / (num_files_NER - files_with_minus_one_NER_AE) if num_files_NER != 0 else 0
average_f1_NER_AE = total_f1_NER_AE / (num_files_NER - files_with_minus_one_NER_AE) if num_files_NER != 0 else 0


print('Number of files for RE is', num_files_RE)

average_recall_RE = total_recall_RE / (num_files_RE-files_with_minus_one_RE) if num_files_RE != 0 else 0
average_precision_RE = total_precision_RE / (num_files_RE-files_with_minus_one_RE) if num_files_RE != 0 else 0
average_f1_RE = total_f1_RE / (num_files_RE-files_with_minus_one_RE) if num_files_RE != 0 else 0

result = {
    "Avg_Rec_NER_DRUG": average_recall_NER_DRUG,
    "Avg_Prec_NER_DRUG": average_precision_NER_DRUG,
    "Avg_F1_score_NER_DRUG": average_f1_NER_DRUG,
    "ErrFilesNum_NER_DRUG": files_with_minus_one_NER_DRUG,
    "Avg_Rec_NER_AE": average_recall_NER_AE,
    "Avg_Prec_NER_AE": average_precision_NER_AE,
    "Avg_F1_score_NER_AE": average_f1_NER_AE,
    "ErrFilesNum_NER_AE": files_with_minus_one_NER_AE,
    "Avg_Rec_RE": average_recall_RE,
    "Avg_Prec_RE": average_precision_RE,
    "Avg_F1_score_RE": average_f1_RE,
    "ErrFilesNum_RE": files_with_minus_one_RE
}

with open(file_dir + "/avg_scores.json", "w") as output:
    json.dump(result, output, indent=4)

print("Evaluation Results:")
for key, value in result.items():
    print(f"{key}:\t {value}")
