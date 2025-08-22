'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import ast
import pandas as pd

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    model_pred_df = pd.read_csv("data/prediction_model_03.csv")
    genres_df = pd.read_csv("data/genres.csv")
    return model_pred_df, genres_df

def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''
    if "genre" in genres_df.columns:
        genre_list = [str(x).strip() for x in genres_df["genre"].dropna().tolist()]
    else:
        s = set()
        for _, row in model_pred_df.iterrows():
            try:
                lst = ast.literal_eval(row["actual genres"])
            except Exception:
                lst = []
            for g in lst:
                if str(g).strip():
                    s.add(str(g).strip())
        genre_list = sorted(s)
    genre_true_counts = {g: 0 for g in genre_list}
    genre_tp_counts = {g: 0 for g in genre_list}
    genre_fp_counts = {g: 0 for g in genre_list}
    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts
