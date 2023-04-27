# NER Evaluation Function
def get_metrics_NER(tpsClassesNER, fpsClassesNER, fnsClassesNER):
    prec_ae = tpsClassesNER['AE'] / (tpsClassesNER['AE'] + fpsClassesNER['AE'])
    rec_ae = tpsClassesNER['AE'] / (tpsClassesNER['AE'] + fnsClassesNER['AE'])
    f1_ae = (2 * prec_ae * rec_ae) / (prec_ae + rec_ae)
    print('AE entity')
    print('Precision: {:.4f}'.format(prec_ae))
    print('Recall: {:.4f}'.format(rec_ae))
    print('F1 Score: {:.4f}'.format(f1_ae))
    print('___________________')

    prec_drug = tpsClassesNER['DRUG'] / (tpsClassesNER['DRUG'] + fpsClassesNER['DRUG'])
    rec_drug = tpsClassesNER['DRUG'] / (tpsClassesNER['DRUG'] + fnsClassesNER['DRUG'])
    f1_drug = (2 * prec_drug * rec_drug) / (prec_drug + rec_drug)
    print('DRUG entity')
    print('Precision: {:.4f}'.format(prec_drug))
    print('Recall: {:.4f}'.format(rec_drug))
    print('F1 Score: {:.4f}'.format(f1_drug))
    print('####################')
    print('####################')

    return {'DRUG': {'Recall': rec_drug,
                     'Precision': prec_drug,
                     'F1 score': f1_drug},
            'AE': {'Recall': rec_ae,
                   'Precision': prec_ae,
                   'F1 score': f1_ae}}

# RC Evaluation Functions
def evaluate_RE(predicted_rel, gold_rel, gold_chunks, predicted_chunks):
    tp = 0
    fn = 0
    fp = 0
    for p_r in predicted_rel:
        drug_flag = 0
        ae_flag = 0
        # Check if the predicted pair is in gold list.
        if p_r in gold_rel:
            # Check if the span and type of name entities are correct.
            for p_c in predicted_chunks:  ###===== chunks = [("DRUG", 0, 1), ("AE", 3, 3)] // list of (chunk_type, chunk_start, chunk_end)
                if p_c in gold_chunks:
                    if p_r[0] == p_c[2]:
                        drug_flag = 1
                    if p_r[1] == p_c[2]:
                        ae_flag = 1

        if (drug_flag) == 1 and (ae_flag == 1):
            tp += 1
        else:
            fp += 1

    for g_r in gold_rel:
        if g_r not in predicted_rel:
            fn += 1

    return tp, fp, fn

def get_metrics_rel(tp, fp, fn):
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    print('Relations:')
    print('Precision: {:.4f}'.format(prec))
    print('Recall: {:.4f}'.format(rec))
    print('F1 Score: {:.4f}'.format(f1))

    return {'Recall': rec,
            'Precision': prec,
            'F1 score': f1}