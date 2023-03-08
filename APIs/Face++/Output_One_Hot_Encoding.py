# -*- coding: utf-8 -*-
"""05_Output_One_Hot_Encoding

This py-file encodes the Face++ API output into a one-hot encoded dataframe
Note:
    The max value of each picture is defined to be the detected emotion.
    If multiple emotions are detected with the same confidence, all are set to 1

"""
# import the required libraries
import sklearn.preprocessing
import pandas as pd
# import the raw data
raw_fer = pd.read_csv('04 AffectNet/03 API Outputs/05 Face++/Facepp_FER.csv', index_col=0)
# filter out results where Google Vision did not detect a face
not_detected = raw_fer.loc[raw_fer['anger'] == 'no face detected']
not_detected.to_csv('04 AffectNet/03 API Outputs/05 Face++/FacePP_Not_detected.csv')
# drop 'no face detected' before one hot encoding
raw_fer = raw_fer[raw_fer['anger'] != 'no face detected']

# ===================================================== ENCODING

# encode the str values returned by Google Vision to numerical values
# no encoding of 'UNKONWN' required as not in dataframe
# in case the Vision CLient was able to detect a face, it was also able to classify the emotion

# Rescaling values to range from 0 to 1
raw_fer.iloc[:,1:] = sklearn.preprocessing.MinMaxScaler().fit_transform(raw_fer.iloc[:,1:])

# set all non-max values to 0 of each column (after transposing the df)
raw_t = raw_fer.T.copy()
raw_t.iloc[1:,:][raw_t.iloc[1:,:]!=raw_t.iloc[1:,:].max()] = 0
# set the remaining non-zero values (maximum) to 1
raw_t.iloc[1:,:][raw_t.iloc[1:,:] != 0] = 1
encoded = raw_t.T

# safte the file to csv
encoded.to_csv('04 AffectNet/03 API Outputs/05 Face++/FacePP_One_Hot_Encoded.csv')
