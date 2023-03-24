# -*- coding: utf-8 -*-
"""04_CloudVision.py

This .py file defines a class VisionClient to instantiate a Cloud Vision Client
and saves the output in .csv files when the API is called.

@main(): Executes the API Post for the AffectNet Sample and saves the
outputs as described below.

Constants to be set:
@WDIR: Path of working directory
@GOOGLE_APPLICATION_CREDENTIALS: Path to be provided to .json file with account credentials
@IMAGES_FOLDER: Path to folder with the images to be processed
@GOOGLE_FER: Path to dedicated output file
@GOOGLE_JSON: Path to dedicated output file

@output: .csv files
    Google_FER == emotions with corresponding confidence for each image
    Google_JSON == raw json output for each image
    Google_OHE == one hot encoded FER output

PREREQUISITES:
    1. Google Cloud Platform set-up
    find details --> https://cloud.google.com/vision/docs/detecting-faces?hl=de
"""
# importing the required modules
import os
import time
from google.cloud import vision
import pandas as pd
import sklearn.preprocessing

#constants to be defined
WDIR = 'PATH_TO_WDIR'
GOOGLE_APPLICATION_CREDENTIALS = 'PATH_TO_GOOGLE_CREDENTIALS'
IMAGES_FOLDER = 'PATH_TO_IMAGES'
GOOGLE_FER = 'PATH_TO_OUTPUT_FER'
GOOGLE_JSON = 'PATH_TO_OUTPUT_JSON'
GOOGLE_NOTDETECTED = 'PATH_TO_CONTROL_FILE'
GOOGLE_OHE = 'PATH_TO_OUTPUT_OHE'

class VisionClient:
    """VisionClient

    use to create instance of a Cloud Vision client

    """
    def __init__(self):
        # Instantiates a client of the Cloud Vision client for Image Annotation
        self.client = vision.ImageAnnotatorClient()

    def detect_lables_google(self, image, path):
        """detect_lables_google(self, image, path)

        Use detect_lables_google to call the face detection method.

        @image: image to be processed astype <DirEntry> originating from fetching
        file name with os.scandir().
        @path: path to the folder containing the file must be specified.

        @return:    a tuple with
            if face is detected:
            [0] == image and emotion likelihoods: [image name, anger, joy, surprise, sorrow]
            [1] == raw json output
            else:
            outputs "no face detected" for the corresponding image

        """
        # reading the image (astype <DirEntry>) as a binary variable (bytes)
        with open(path+image.name, 'rb') as image_file:
            img_byte = image_file.read()
        # calling the face detection API with the image input as a vision.Image instance
        response = self.client.face_detection(image = vision.Image(content=img_byte))
        # try statement executed if a face is being detected
        try:
            # get only face with highest detection confidence (AffectNet labelled as such)
            # can trigger IndexError if no face is detected
            face = response.face_annotations[0]
            # Names of likelihood from google.cloud.vision.enums
            likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                               'LIKELY', 'VERY_LIKELY')
            return ([image.name,\
                    likelihood_name[face.anger_likelihood],\
                    likelihood_name[face.joy_likelihood],\
                    likelihood_name[face.surprise_likelihood],\
                    likelihood_name[face.sorrow_likelihood]], [image.name, face])
        # Error handling in case the Google Vision API does not detect a face
        except IndexError:
            return ([image.name,\
                    "no face detected", "no face detected",\
                    "no face detected", "no face detected"], [image.name, "no face detected"])

def main():
    """main()

    Executed when run from the command line, defining the operational functionality
    of the script while excluding potentially resource heavy functions.

    """
    print('04_CloudVision exe')
    start = time.time()
    # specifying the working directory
    os.chdir(WDIR)
    # required definition of credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= GOOGLE_APPLICATION_CREDENTIALS
    # initiating two Pandas Data Frames as containers for csv files.
    # @google_fer as an easy-to-use container with the image name and the corresponding
    # emotion likelihoods as detected by Cloud Vision.
    google_fer = pd.DataFrame(columns=\
                              ['picture','anger', 'joy', 'surprise', 'sorrow'])
    # @google_json as a container for the raw data with the corresponding image name
    # as well as the time to excecute the file in row index [-1]
    google_json = pd.DataFrame(columns=\
                              ['picture', 'response'])
    # creating instance of VisionClient
    client = VisionClient()
    # iterating over the specified folder with the images to be processed.
    for image in os.scandir(IMAGES_FOLDER):
        if image.name != '.DS_Store': # excludes generated file (by macOS)
            # running the detect_lables_google func to call the API and storing the
            # outputs in the predefined containers
            label_count = client.detect_lables_google(image, IMAGES_FOLDER)
            google_fer.loc[len(google_fer.index)] = label_count[0]
            google_json.loc[len(google_json.index)] = label_count[1]
            # tracking progress of processed images
            print(len(google_fer.index))
        else:
            continue
    # add row with time used to execute script
    google_json.loc[len(google_json.index)] = ["time to execute: ", time.time() - start]
    
    # extract images where no face was detected
    not_detected = google_fer.loc[google_fer['joy'] == 'no face detected']
    # drop 'no face detected' before one hot encoding
    google_fer['no face detected'] = pd.Series(google_fer['joy'] == 'no face detected').astype(int)
    raw_fer = google_fer.replace('no face detected', 0)
    
    
    # ===================================================== ONE HOT ENCODING
    
    # encode the str values returned by Google Vision to numerical values
    # no encoding of 'UNKONWN' required as not in dataframe
    # in case the Vision CLient was able to detect a face, it was also able to classify the emotion
    emotions_list = ['VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY']
    for i, name in enumerate(emotions_list):
        raw_fer.iloc[:,1:] = raw_fer.iloc[:,1:].replace([name], i)
    
    # Rescaling values to range from 0 to 1
    raw_fer.iloc[:,1:-1] = sklearn.preprocessing.MinMaxScaler().fit_transform(raw_fer.iloc[:,1:-1])
    
    # set all non-max values to 0 of each column (after transposing the df)
    raw_t = raw_fer.T.loc[:,:]
    raw_t.iloc[1:-1,:][raw_t.iloc[1:-1,:]!=raw_t.iloc[1:-1,:].max()] = 0
    # set the remaining non-zero values (maximum) to 1
    raw_t.iloc[1:-1,:][raw_t.iloc[1:-1,:] != 0] = 1
    encoded = raw_t.T
    
    encoded = encoded.rename(columns={'joy': 'happy', 'sorrow': 'sad'})
    
    encoded = encoded.reindex(columns=['picture', 'happy', 'sad', 'surprise', 'fear', 'disgust',
                                       'anger', 'contempt', 'neutral', 'confused', 'no face detected'],
                              fill_value = 0)
    no_em_dect = encoded.iloc[:,1:].any(1).astype(int)
    encoded.insert(10, 'no emotions detected', ~no_em_dect+2)

    # saving the outputs fromn the API call to .csv files
    google_fer.to_csv(GOOGLE_FER)
    google_json.to_csv(GOOGLE_JSON)
    encoded.to_csv(GOOGLE_OHE)
    not_detected.to_csv(GOOGLE_NOTDETECTED)
    
    print('\n04_CloudVision exe success')

if __name__ == "__main__":
    main()
