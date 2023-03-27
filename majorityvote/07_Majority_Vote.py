# -*- coding: utf-8 -*-
"""07_Majority_Vote.py

This .py file instantiates and optimises the majority vote of the four APIs as outlined in the 
Main paper. 

constants to be defined:
@WDIR: Path of working directory
@PATH_GOOGLE_OHE: Path to the output data OHE from Google
@PATH_AWS_OHE = Path to the output data OHE from AWS
@PATH_FPP_OHE = Path to the output data OHE from FPP
@PATH_SB_OHE = Path to the output data OHE from SB
@PATH_SAMPLE_OHE = Path to the OHE sample data


@output: displays optimal weights and minimal loss in the console and majority_vote.csv saved with the corresponding data 

PREREQUISITES:
    1. Prepared AffectNet data set
    2. All requests success for the four APIs

"""

# import libraries
import os 
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.optimize import minimize
import numpy as np
import pandas as pd 

#set constants
WDIR = 'PATH_TO_WDIR'
PATH_GOOGLE_OHE = 'PATH_TO_GOOGLE_OHE'
PATH_AWS_OHE = 'PATH_TO_AWS_OHE'
PATH_FPP_OHE = 'PATH_TO_FPP_OHE'
PATH_SB_OHE = 'PATH_TO_SB_OHE'
PATH_SAMPLE_OHE = 'PATH_TO_SAMPLE_OHE'

# load data 

os.chdir(WDIR)

google = pd.read_csv('PATH_GOOGLE_OHE', index_col= 0)
aws = pd.read_csv('PATH_AWS_OHE', index_col = 0)
fpp = pd.read_csv('PATH_FPP_OHE', index_col = 0)
sb = pd.read_csv('PATH_SB_OHE', index_col = 0)
Sample_ohe = pd.read_csv('PATH_SAMPLE_OHE', index_col=0)
df_lst = [Sample_ohe, google, aws, fpp, sb]
for df in df_lst: 
    df.iloc[:,0] = df.iloc[:,0].str.replace('.jpg', '')
    df.iloc[:,0] = df.iloc[:,0].str.replace('val', '').astype(int)

# reset index by picture id
Sample_ohe = Sample_ohe.set_index('picture')
google = google.set_index('picture')
aws = aws.set_index('picture')
fpp = fpp.set_index('picture')
sb = sb.set_index('picture')

# change column order
google = google.reindex(['happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt'], axis = 'columns', fill_value = 0)
aws = aws.reindex(['happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt'], axis = 'columns', fill_value = 0)
fpp = fpp.reindex(['happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt'], axis = 'columns', fill_value = 0)
sb = sb.reindex(['happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt'], axis = 'columns', fill_value = 0)

# sort the data
Sample_ohe = Sample_ohe.sort_index()
google = google.sort_index()
aws = aws.sort_index()
fpp = fpp.sort_index()
sb = sb.sort_index()

# method to retrieve the majority vote
def get_majority_vote(df1, df2, df3, df4, weights):
    # Combine the dataframes
    votes = df1*weights[0] + df2*weights[1] + df3*weights[2] + df4*weights[3]
    majority_vote_df = votes.eq(votes.where(votes != 0).max(1), axis=0).astype(int)
    
    return majority_vote_df

# method to optimise the API weights using cohen's kappa as outlined in the main paper
def optimize_weights(df1, df2, df3, df4, true_labels):
    # Define the error function
    def error(weights):
        # Get the predicted labels
        predicted_labels = get_majority_vote(df1, df2, df3, df4, weights)
        
        sample_cm_lables = true_labels.loc[:,['happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']]
        #sample_cm_lables = sample_cm_lables.loc[~(sample_cm_lables==0).all(axis=1)]
        pred_ohe_pl = predicted_labels.loc[:,['happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']]
        pred_ohe_pl = pred_ohe_pl.loc[list(sample_cm_lables.index),:]
        true = np.argmax(np.array(sample_cm_lables), axis = 1)
        pred_labels = np.argmax(np.array(pred_ohe_pl), axis = 1)
        conf_matrix = confusion_matrix(true, pred_labels)
        conf_matrix_df = pd.DataFrame(conf_matrix,
                                      columns = ['happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt'],
                                      index = ['happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt'])
	
	#define metrics to derive cohen's kappa
        fp = conf_matrix_df.sum(axis=0) - np.diag(conf_matrix_df) 
        fn = conf_matrix_df.sum(axis=1) - np.diag(conf_matrix_df)
        tp = pd.Series(np.diag(conf_matrix_df), index=conf_matrix_df.index)
        tn = conf_matrix_df.to_numpy().sum() - (fp + fn + tp)
        
        
        fp = pd.Series(fp.astype(float))
        fn = pd.Series(fn.astype(float))
        tp = pd.Series(tp.astype(float))
        tn = pd.Series(tn.astype(float))

        pp_comp = tp + fp
        p_comp = tp + fn

        N = tp.sum()+fp.sum()
        c = tp.sum()
        sumproduct_pp_tp = sum(pp_comp*p_comp)
        kappa = (c*N-sumproduct_pp_tp)/(N**2-sumproduct_pp_tp)
        
        res_lst.append((kappa, weights))

        #print(f'{kappa=}, {weights=}')
    
        
        return 1-kappa # We want to minimize the error, so we return 1 - F1
    
    # Define the optimization function
    def objective(weights):
        return error(weights)

    # Set the initial weights
    initial_weights = [0.26, 0.25, 0.25, 0.24]
    
    global res_lst
    res_lst = []

    # Optimize the weights using gradient descent
    res = minimize(objective, initial_weights, method = 'BFGS', jac = '2-point', options={'disp': True, 'finite_diff_rel_step': 0.01 })
    
    return res.x

# Find the optimal weights
optimal_weights = optimize_weights(google, fpp, aws, sb, Sample_ohe)
print(optimal_weights/ sum(optimal_weights)*1)
print(sorted(res_lst, key = lambda x: x[0], reverse=True)[:2])

votes = google*optimal_weights[0] + fpp*optimal_weights[1] + aws*optimal_weights[2] + sb*optimal_weights[3]
majority_vote_df = votes.eq(votes.where(votes != 0).max(1), axis=0).astype(int)
majority_vote_df.to_csv('01 Analytics/Majority Vote/majority_vote.csv')
    
    
    
