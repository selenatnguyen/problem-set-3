'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''

    for idx,row in model_pred_df.iterrows():
        this_genres = eval(row["actual genres"])
    
        for true_g in this_genres:
            genre_true_counts[true_g] = genre_true_counts.get(true_g, 0) + 1
            
        pred_g = row["predicted"]
        if pred_g in this_genres:
            genre_tp_counts[pred_g] = genre_tp_counts.get(pred_g, 0) + 1
        else:
            genre_fp_counts[pred_g] = genre_fp_counts.get(pred_g, 0) + 1

    tp = 0
    fp = 0
    fn = 0
    for genre in genre_list:
        tp += genre_tp_counts[genre]
        fp += genre_fp_counts[genre]
        fn += (genre_true_counts[genre] - genre_tp_counts[genre])
        
    micro_prec = tp / (tp + fp)
    micro_recall = tp / (tp + fn)
    micro_f1 = (2 * (micro_prec * micro_recall) / (micro_prec + micro_recall))

    macro_prec_list = []
    macro_recall_list = []
    macro_f1_list = []

    for genre in genre_list:
        local_tp = genre_tp_counts[genre]
        local_fp = genre_fp_counts[genre]
        local_fn = (genre_true_counts[genre] - genre_tp_counts[genre])
        
        local_prec = 0.0
        local_recall = 0.0
        
        if local_tp > 0:
            local_prec = local_tp / (local_tp + local_fp)
            local_recall = local_tp / (local_tp + local_fn)
        local_f1 = 0.0
        if (local_prec + local_recall) > 0:
            local_f1 = (2 * (local_prec * local_recall) / (local_prec + local_recall))
            
        macro_prec_list.append(local_prec)
        macro_recall_list.append(local_recall)
        macro_f1_list.append(local_f1)
    
    return micro_prec, micro_recall, micro_f1, macro_prec_list, macro_recall_list, macro_f1_list

def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''

    pred_rows = []
    true_rows = []

    for idx,row in model_pred_df.iterrows():
        this_genres = eval(row["actual genres"])
        pred_g = {row["predicted"]}

        true_rows.append({
            g:1 if g in this_genres else 0 for g in genre_list
        })
        
        pred_rows.append({
            g:1 if g in pred_g else 0 for g in genre_list
        })
        
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)

    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(true_matrix, pred_matrix, average="macro")
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(true_matrix, pred_matrix, average="micro")
    return macro_prec, macro_rec, macro_f1, micro_prec, micro_rec, micro_f1
