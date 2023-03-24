# -*- coding: utf-8 -*-
"""01_AffectNetSample.py

This .py file cleans the data set in order to filter out unwanted pictures.
In the case of this analysis all "Neutral" are excluded as only the distinct
emotions are relevant. Further, a sample of a 1000 pictures is randomly
created. This sample retains the original proportion of each emotion in
the original complete data set.

@main(): executes the sample generation for a defined set of images and a 
determined sample size.

constants to be definded:
@SAMPLE_SIZE: sample size
@PATH_SAMPLE_FOLDER: Path to folder with images to be processed


@output: .csv files
    Amazon_FER == two most likely emotions with corresponding confidence for each image
    Amazon_JSON == raw json ouput for each image
    AWS_One_Hot_Encoded == One hot encoded output

PREREQUISITES:
    1. Access to data set (here: http://mohammadmahoor.com/affectnet/)
"""
# importing necessary libraries$

import os
import random
import shutil
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


def file_label_extraction():
    print('file extraction started')
    # set-up empty dictionary to extract an Indextype from filename with corresponding emotion.
    # 0-7 mannually labled pictures
    lst = []
    # extraction of filename (index) and corresponding label (emotion).
    for directory in ['train_set/annotations/', 'val_set/annotations/']:
        directory_files = list(filter(lambda x: '_exp' in x, os.listdir(directory)))
        print(directory)
        for filename in directory_files:
            uid = filename if directory == 'train_set/annotations/' else filename +'val'
            lst.append([uid, int(np.load(directory + filename))])
    index_df = pd.DataFrame(lst, columns=['picture', 'label'])
    index_df.to_csv(r'PATH_TO_FILE')
    print('filename extraction complete')

    return index_df

def sample_genertion(sample_size, index_df, path_sample_folder):
    # remove all values in dictionary equal to 0 = neutral to only retain distinct emotions
    cleaned_df = index_df[index_df['label'] != 0]
    cleaned_dict = dict(zip(cleaned_df['picture'], cleaned_df['label']))

    # count the instances of each emotion.
    # control line: Count is given in the AffectNet summary paper
    emotions_counter = Counter(cleaned_dict.values())

    # calculate relative amount of each emotion in terms of desired sample size
    sample_counter = [round(emotions_counter[x]/len(cleaned_dict)*sample_size) for x in range(1,8)]

    # empty dictionary to store the final randomized indizes of the pictures
    extract_index_dict = {}
    extract_df = pd.DataFrame(columns=['picture', 'label'])

    # randomly sample picture indizes while preserving the initial proportion of emotions
    for i in range(len(sample_counter)):
        indexlist = [k for k, v in cleaned_dict.items() if v == i+1]
        randomlist = random.sample(indexlist, k = sample_counter[i])
        extract_index_dict[i+1] = randomlist
        extract_df = pd.concat([extract_df, pd.DataFrame({'picture':randomlist})])
        extract_df = extract_df.fillna(i+1)

    # One Hot Encoding of the sample labels
    extract_np = np.asarray(extract_df.iloc[:,1])
    integer_encoded = extract_np.reshape(len(extract_np), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    final_df = pd.DataFrame(columns=['picture', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt'])
    final_df['picture'] = extract_df.iloc[:,0].str.replace('_exp.npy', '').astype(str) + '.jpg'
    final_df.iloc[:,1:] = onehot_encoded     
    
    final_df.to_csv('PATH_TO_FILE')


    for i in extract_index_dict.values():
        for name in i:
            if name.endswith('val'):
                shutil.copyfile('val_set/images/'+ name[:-11] +'.jpg',\
                                path_sample_folder + name[:-11] +'.jpg')
            else:
                shutil.copyfile('train_set/images/'+name[:-8] +'.jpg',\
                                path_sample_folder + name[:-8] +'.jpg')

    print('data sucessfully sampled')

def main():
    # setting the working directory to the AffectNet Folder
    os.chdir('WDIR')
    
    # determine Sample Size
    SAMPLE_SIZE = 1000
    # determine the folder of the final sample (note: raw string required)
    PATH_SAMPLE_FOLDER = r'PATH_TO_AFFECTNET_FOLDER'
    random.seed(10)
    # execute file_lable_extraction and the sample_generation
    sample_genertion(SAMPLE_SIZE, file_label_extraction(), PATH_SAMPLE_FOLDER)

if __name__ == "__main__":
    main()
    
