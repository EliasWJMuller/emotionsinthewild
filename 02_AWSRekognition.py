# -*- coding: utf-8 -*-
"""02_AWSRekognition.py

This .py file defines a class RekognitionClient to instantiate a AWS Rekognition Client
and saves the output in .csv files when the API is called.

@main(): excecutes the API Post for the AffectNet Sample and saves the
outputs as described below.

constants to be definded:
@WDIR: Path of working directory
@IMAGES_FOLDER: Path to folder with images to be processed
@AMAZON_FER: Path to dedicated output file
@AMAZON_JSON: Path to dedicated output file
@AMAZON_OHE: Path to dedicated output file

@output: .csv files
    Amazon_FER == two most likely emotions with corresponding confidence for each image
    Amazon_JSON == raw json ouput for each image
    AWS_One_Hot_Encoded == One hot encoded output

PREREQUISITES:
    1. AWS IAM Account with access to AWS Rekognition
    2. AWS CLI and AWS SDKs set-up (configure credentials)
    find details --> https://docs.aws.amazon.com/rekognition/latest/dg/getting-started.html
"""
# importing the required modules
import os
import time
import boto3
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


#defining constants
WDIR = 'WDIR_PATH'
IMAGES_FOLDER = 'IMAGES_PATH'
AMAZON_FER = 'PATH_TO_OUTPUT_FER'
AMAZON_JSON = 'PATH_TO_OUTPUT_JSON'
AMAZON_OHE = 'PATH_TO_OUTPUT_OHE'

class RekognitionClient:
    """RekognitionClient

    instantiates a AWS Rekognition client

    """
    def __init__(self):
        # initializing an AWS client with the 'rekognition' module
        self.client = boto3.client('rekognition')

    def detect_lables_amazon(self, photo):
        """detect_lables_amazon(self, photo)

        Use detect_lables_amazon to call the Amazon Rekognition API for applicable images.

        This function leverages the AWS specific boto3 SDK.

        @photo:  input file of image to be processed astype <DirEntry> originating from
        fetching the image name with os.scandir().

        @return:    a tuple with
            [0] == 1st emotion detected: [Emotion, Confidence]
            [1] == 2nd emotion detected: [Emotion, Confidence]
            [2] == raw json output

        """
        # reading the image (astype <DirEntry>) as a binary variable and calling the
        # detect_faces method of the initialized boto3 AWS client. 'detect_faces'
        # returns a dict of faces found in the processed image with the attributes specified
        # @params: Bytes(string): image bytes; Attributes(string): ['ALL', 'DEFAULT']
        with open(photo, 'rb') as image:
            response = self.client.detect_faces(Image={'Bytes': image.read()}, Attributes = ['ALL'])
        return (list(response['FaceDetails'][0]['Emotions'][0].values()),\
                list(response['FaceDetails'][0]['Emotions'][1].values()),\
                response)

def main():
    """main()

    Executed when run from the command line, defining the operational functionality
    of the script while excluding potentially ressource heavy functions.

    """
    print('02_AmazonRequest exe')
    # specifying the working directory
    os.chdir(WDIR)
    start = time.time()
    # initiating two Pandas Data Frames as containers for csv files.
    # @amazon_fer as an easy-to-use container with the image name and the corresponding
    # two most likely displayed emotions as detected by AWS Rekognition and confidence.
    amazon_fer = pd.DataFrame(columns=\
                              ['picture','1st emotion', 'confidence', '2nd emotion', 'confidence'])
    # @amazon_json as a container for the raw data with the corresponding image name
    # as well as the time to excecute the file in row index [-1]
    amazon_json = pd.DataFrame(columns=\
                              ['picture', 'response'])
    # creating instance of AWS Rekognition
    client = RekognitionClient()
    # iterating over the specified folder with the images to be processed.
    for image in os.scandir(IMAGES_FOLDER):
        if image.name != '.DS_Store': # excludes generated file (by macOS)
            # running the detect_lables_amazon func to call the API and storing the
            # outputs in the predefined containers
            labels = client.detect_lables_amazon(image)
            amazon_fer.loc[len(amazon_fer.index)] = [image.name,\
                                                     labels[0][0], labels[0][1],\
                                                     labels[1][0], labels[1][1]]
            amazon_json.loc[len(amazon_json.index)] = [image.name, labels[2]]
            # tracking progress of processed images
            print(len(amazon_fer.index))
        else:
            continue
    # add row with time used to excute script
    amazon_json.loc[len(amazon_json.index)] = ["time to excecute: ", time.time() - start]
    # ===================================================== ONE HOT ENCODING

    # One Hot Encoding of the sample labels
    extract_np = np.asarray(amazon_fer.iloc[:,1])
    integer_encoded = extract_np.reshape(len(extract_np), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    encoded = pd.DataFrame(columns=['picture', 'anger', 'neutral', 'confused', 'disgust',
                                    'fear', 'happy', 'sad', 'surprise'])
    encoded['picture'] = amazon_fer.iloc[:,0]
    encoded.iloc[:,1:] = onehot_encoded
    encoded.reindex(columns=['picture', 'happy', 'sad', 'surprise', 'fear',
                             'disgust', 'anger', 'contempt', 'neutral', 'confused'],
                    fill_value = 0)
    no_em_dect = encoded.iloc[:,1:].any(1).astype(int)
    encoded['no emotions detected'] = ~no_em_dect+2
    encoded['no face detected'] = 0
    # no_f_dect = encoded.iloc[:,1:].any(1).astype(int)
    # encoded['no face detected'] = ~no_f_dect
    
    # saving the outputs fromn the API call to .csv files
    encoded.to_csv(AMAZON_OHE)
    amazon_fer.to_csv(AMAZON_FER)
    amazon_json.to_csv(AMAZON_JSON)
    print('\n02_AmazonRequest exe success')

# basic guard to protect from accidental excecution of the script
if __name__ == "__main__":
    main()
