# -*- coding: utf-8 -*-
"""06_Analytics.py

This .py file creates the analytics based on the outlined performance metrics in Chapter 04 (Results) of the underlying paper.

constants to be definded:
@WDIR: Path of working directory
@FILE_PATH: Path for files to be saved to 


@output: .csv files with the analytics for each API (uncomment the respective code block for each API at the bottom of this code)

PREREQUISITES:
    1. All requests success for the four APIs
    2. Majority vote completed (07_Majority_Vote.py)

"""

import os
import pandas as pd
import numpy as np
import sklearn.metrics
import csv
import matplotlib.pyplot as plt


WDIR = 'PATH_TO_WDIR
FILE_PATH = 'PATH_TO_SAVE'


#Data Preparation 
#load all the data gathered from the request files

os.chdir(WDIR)
Sample_ohe = pd.read_csv('PATH_TO_OHE_FILE', index_col=0)
Sample_ohe = Sample_ohe.reindex(columns=['picture', 'happy', 'sad', 'surprise', 'fear',
                         'disgust', 'anger', 'contempt','neutral', 'confused', 'no emotions detected','no face detected'],
                fill_value = 0)
AWS_ohe = pd.read_csv('PATH_TO_OHE_FILE', index_col=0)
AWS_ohe = AWS_ohe.reindex(columns=['picture', 'happy', 'sad', 'surprise', 'fear',
                         'disgust', 'anger', 'contempt','neutral', 'confused', 'no emotions detected','no face detected'],
                fill_value = 0)
Google_ohe = pd.read_csv('PATH_TO_OHE_FILE', index_col=0)
SB_ohe = pd.read_csv('PATH_TO_OHE_FILE', index_col=0)
FPP_ohe = pd.read_csv('PATH_TO_OHE_FILE', index_col=0)

M_vote = pd.read_csv('PATH_TO_OHE_FILE')


#modify all the indexes to align 
df_list = [Sample_ohe, AWS_ohe, Google_ohe, SB_ohe, FPP_ohe]
for df in df_list: 
    df.iloc[:,0] = df.iloc[:,0].str.replace('.jpg', '')
    df.iloc[:,0] = df.iloc[:,0].str.replace('val', '').astype(int)


#sort data df
Sample_ohe = Sample_ohe.set_index('picture').sort_index()
AWS_ohe = AWS_ohe.set_index('picture').sort_index()
Google_ohe = Google_ohe.set_index('picture').sort_index()
SB_ohe = SB_ohe.set_index('picture').sort_index()
FPP_ohe = FPP_ohe.set_index('picture').sort_index()
M_vote = M_vote.set_index('picture').sort_index()

class Analytics(object):
    """Analytics(object)
       Class dedicated to handling all analytics. When Initialising creates the required confusion matrices for given data 
       On which the metrics can be computed.

       @df_sample: takes a give API output
       @df_pred: takes the truth labels
       @labels: takes the labels for which to compute the metrics
       @provider (str): provider name
    """
    def __init__(self, df_sample, df_pred, labels, provider):
        self.sample = df_sample
        self.pred = df_pred
        self.provider = provider
        sample_cm_lables = self.sample.loc[:,labels]
        #sample_cm_lables = sample_cm_lables.loc[~(sample_cm_lables==0).all(axis=1)]
        pred_ohe_pl = self.pred.loc[:,labels]
        pred_ohe_pl = pred_ohe_pl.loc[list(sample_cm_lables.index),:]
        
        self.sample_pl = np.argmax(np.array(sample_cm_lables), axis = 1)
        self.pred_pl = np.argmax(np.array(pred_ohe_pl), axis = 1)
        conf_matrix = sklearn.metrics.confusion_matrix(self.sample_pl, self.pred_pl)
        self.conf_matrix_df = pd.DataFrame(conf_matrix, columns = labels, index = labels)

    def add_matrix_totals(self):
	# add the total columns
        self.conf_matrix_df['sample total'] = self.conf_matrix_df.sum(axis =1)
        self.conf_matrix_df.loc['prediction total',:] = self.conf_matrix_df.sum(numeric_only=True, axis=0)

    def confusion_matrix(self, col_labels = None, row_labels = None, mode = ''):
        print(self.provider)
        if col_labels == None: 
            col_labels = ['happy', 'sad', 'surprise', 'fear', 'disgust', 'anger',
             'contempt','neutral', 'confused', 'no emotions detected', 'no face detected',
             'sample total']
        else:
            col_labels.extend(['no emotions detected', 'no face detected', 'sample total'])
        
        if row_labels == None:
            row_labels = ['happy', 'sad', 'surprise', 'fear', 'disgust', 'anger',
             'contempt', 'prediction total']
        
        self.conf_matrix_df = self.conf_matrix_df.reindex(columns=col_labels,
                                                          fill_value = 0)
        self.conf_matrix_df = self.conf_matrix_df.reindex(row_labels,
                                                          fill_value = 0)
        
        self.conf_matrix_df.to_csv(FILE_PATH+'Confusion Matrices/'+self.provider+mode+'.csv')
        
        analytics_matrix = self.conf_matrix_df.reindex(col_labels[:-1],
                                                       fill_value = 0)
        analytics_matrix.drop(columns=['sample total'], inplace=True)
        
        fp_comp = analytics_matrix.sum(axis=0) - np.diag(analytics_matrix) 
        fn_comp = analytics_matrix.sum(axis=1) - np.diag(analytics_matrix)
        tp_comp = pd.Series(np.diag(analytics_matrix), index=analytics_matrix.index)
        tn_comp = analytics_matrix.to_numpy().sum() - (fp_comp + fn_comp + tp_comp)
        
        fp_comp = pd.Series(fp_comp.astype(float))
        fn_comp = pd.Series(fn_comp.astype(float))
        tp_comp = pd.Series(tp_comp.astype(float))
        tn_comp = pd.Series(tn_comp.astype(float))

        p_comp = tp_comp + fn_comp
        n_comp = fp_comp + tn_comp
        pp_comp = tp_comp + fp_comp
        pn_comp = fn_comp + tn_comp


        # compute metrics overall 
        statistics_dict_all = self.statistics_dict(fp_comp.sum(), fn_comp.sum(),
                                                   tp_comp.sum(), tn_comp.sum(),
                                                   p_comp.sum(), n_comp.sum(),
                                                   pp_comp.sum(), pn_comp.sum())
                
        with open(FILE_PATH+'Statistics/'+self.provider+'/'+f'overall{mode}_stats.csv','w') as f:
            w = csv.writer(f)
            w.writerows(statistics_dict_all.items())
        
	# compute metrics for each emotion individually
        for i, label in enumerate(fp_comp):
            
            fp, fn, tp, tn, p, n, pp, pn = fp_comp[i], fn_comp[i], tp_comp[i],\
                tn_comp[i], p_comp[i], n_comp[i], pp_comp[i], pn_comp[i]
            
            statistics_dict = self.statistics_dict(fp, fn, tp, tn, p, n, pp, pn)
            
            emotion = fp_comp.index[i]
            
            with open(FILE_PATH+'Statistics/'+self.provider+'/'+f'{emotion}{mode}_stats.csv','w') as f:
                w = csv.writer(f)
                w.writerows(statistics_dict.items())

        return self.conf_matrix_df, analytics_matrix
    
    def statistics_dict(self, fp, fn, tp, tn, p, n, pp, pn):
        p1 = (tp+fn)/(tp+fp+tn+fn)
        p2 = (tp+fp)/(tp+fp+tn+fn)
        random_accuracy = (p1*p2)+(1-p1)*(1-p2) 
        statistics_dict = {   
        'accuracy' : (tp+tn)/(p+n),
        'precision' : tp/pp,
        'recall' : tp/p,
        'F1_score' : (2*((tp/pp)*(tp/p)))/((tp/pp)+(tp/p)),
        'true pos rate' : tp/(tp+fn),
        'false neg rate' : fn/(tp+fn),
        'false pos rate' : fp/(fp+tn),
        'true neg rate' : tn/(tn+fp),
        'cohens kappa': ((tp+tn)/(p+n)-random_accuracy)/(1-random_accuracy)
        }
        
        return statistics_dict 
        

    def normalize_matrix(self, matrix, mode = ''):
	# gives the option to normalise the confusion matrices
        matrix = matrix.iloc[:-1,:-1].div(matrix.iloc[:-1,:-1].sum(axis=1), axis=0)
        matrix.to_csv(FILE_PATH+self.provider+mode+'_norm'+'.csv')
        return matrix

"""
Each snippet below is for one of the four APIs but calculates the complete set of statistics according to 
EmSet1 (complete). Adapt the labels to include to calculate for EmSet2 and EmSet3.

"""

"""
#===========================    GOOGLE VISION    ===========================

#------ PROVIDED LABELS IMG
google_labels = ['happy', 'sad', 'surprise', 'fear', 'disgust', 'anger',
             'contempt', 'no emotions detected', 'no face detected'] 
google_detectable = ['happy', 'sad', 'surprise','anger'] 
google_base = Analytics(Sample_ohe, Google_ohe, google_labels, 'google')
google_base.add_matrix_totals()

#-----
google_calc = google_base.confusion_matrix()
google_cm = google_calc[0]
google_control = google_calc[1]
google_cm_norm = google_base.normalize_matrix(google_cm)

#===========================    AWS REKOGNITION    ===========================

#------ PROVIDED LABELS IMG
aws_labels = ['happy', 'sad', 'surprise', 'fear', 'disgust', 'anger',
             'contempt', 'neutral', 'confused']
aws_detectable = ['happy', 'sad', 'surprise', 'fear', 'disgust', 'anger',
                  'neutral', 'confused']
aws_base = Analytics(Sample_ohe, AWS_ohe, aws_labels, 'aws')
aws_base.add_matrix_totals()

#-----
aws_calc = aws_base.confusion_matrix()
aws_cm = aws_calc[0]
aws_statistics = aws_calc[1]
aws_cm_norm = aws_base.normalize_matrix(aws_cm)

#===========================    SB API    ===========================

#------ PROVIDED LABELS IMG
sb_labels = ['happy', 'sad', 'surprise', 'fear', 'disgust', 'anger',
             'contempt', 'neutral', 'no emotions detected', 'no face detected'] 
sb_detectable = ['happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'neutral'] 
sb_base = Analytics(Sample_ohe, SB_ohe, sb_labels, 'sb')
sb_base.add_matrix_totals()

#-----
sb_calc = sb_base.confusion_matrix()
sb_cm = sb_calc[0]
sb_cm_norm = sb_base.normalize_matrix(sb_cm)

#===========================    FACE++ API    ===========================

#------ PROVIDED LABELS IMG
fpp_labels = ['happy', 'sad', 'surprise', 'fear', 'disgust', 'anger',
             'contempt', 'neutral', 'no face detected'] 
fpp_detectable = ['happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'neutral'] 
fpp_base = Analytics(Sample_ohe, FPP_ohe, fpp_labels, 'fpp')
fpp_base.add_matrix_totals()

#-----
fpp_calc = fpp_base.confusion_matrix()
fpp_cm = fpp_calc[0]
fpp_statistics = fpp_calc[1]
fpp_cm_norm = fpp_base.normalize_matrix(fpp_cm)
"""