# -*- coding: utf-8 -*-
"""05_FaceppRequest.py

This .py file creates a Detect Client for the Face++ API
and saves the output in .csv files when the API is called.

@FacePPClient(): class creating the required form data and excecuting the request

@main(): executes the API Post for the AffectNet Sample and saves the
outputs as described below.

constants to be defined:
@WDIR: Path of working directory
@IMAGES_FOLDER: Path to folder with images to be processed
@FACEPP_FER: Path to dedicated output file
@FACEPP_JSON: Path to dedicated output file
@FACEPP_NOTDETECTED: Path to dedicated output file
@FACEPP_OHE: Path to dedicated output file
@FACEPP_KEY: SkyBiometry credentials
@FACEPP_SECRET: SkyBiometry credentials

@output: .csv files
    Facepp_FER == emotions with corresponding confidence for each image
    Facepp_JSON == raw json ouput for each image
    FacePP_Not_detected == Extract for faces with no detected face
    FacePP_One_Hot_Encoded == One hot encoded output

PREREQUISITES:
    1. Face++ Account SetUp
    2. Client Authentication (api_key and api_secret)

"""
# importing the required modules
import os.path
import urllib
import time
import base64
import json
import pandas as pd
import sklearn.preprocessing

#constants to be defined
WDIR = 'PATH_TO_WDIR'
IMAGES_FOLDER = 'PATH_TO_IMAGES'
FACEPP_FER = 'PATH_TO_OUTPUT_FER'
FACEPP_JSON = 'PATH_TO_OUTPUT_JSON'
FACEPP_NOTDETECTED = 'PATH_TO_OUTPUT_CONTROL_FILE'
FACEPP_OHE = 'PATH_TO_OUTPUT_OHE'
FACEPP_KEY = 'FACEPP_KEY'
FACEPP_SECRET = 'FACEPP_SECRET'

# DO NOT MODIFY
API_URL = 'https://api-us.faceplusplus.com/facepp/v3/detect'

class FacePPClient(object):
    
    def __init__(self, api_key, api_secret):
        self.key = FACEPP_KEY
        self.secret = FACEPP_SECRET
        self.url = API_URL
        
    def faces_detect(self, image, path): 
        with open(path+image, 'rb') as image_file:
            img = image_file.read()
            img_64 = base64.b64encode(img)
        attributes ='gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,beauty,mouthstatus,eyegaze,skinstatus'
        # no need to specify 'calculate all' for faces in image:
        # automatic hierarchy defined to list the face with greatest bounding box first
        data = urllib.parse.urlencode({"api_key" : self.key, "api_secret": self.secret, "image_base64": img_64, "return_attributes": attributes }).encode()
        #buld http request
        req=urllib.request.Request(self.url, data=data)
        #post data to server
        resp = urllib.request.urlopen(req, timeout=10)
        #get response
        response=resp.read().decode('utf-8')
        # convert to dictionary
        
        response_data = self.generate_response(image, response) 
        return response_data
    
    def generate_response(self, image, json_response):
        """generate_response(self, image_name, json_response)

        Use send_request to call the SkyBiometry API for applicable images.

        @image_name: name of input image
        @json_response: API response to be processed

        @return: Output depending on API response

        """
        resp = json.loads(json_response)
        try:
            faces = resp['faces']
            if len(faces) == 0: 
                return ([image, 'no face detected', 'no face detected', 'no face detected', 'no face detected', 'no face detected',
                         'no face detected', 'no face detected', 'no face detected', 'no face detected'],
                            [image, json_response])
            else:
                face = faces[0]['attributes']['emotion']
                return ([image, face['happiness'], face['sadness'], face['surprise'], face['fear'], 
                         face['disgust'], face['anger'], 0, face['neutral'], 0],
                        [image, resp])
        except: 
            print(resp, image)
def main():
    """main()

    Executed when run from the command line, defining the operational functionality
    of the script while excluding potentially resource heavy functions.

    """
    print('05_FaceppRequest exe')
    start = time.time()
    # specifying the working directory
    os.chdir(WDIR)
    # initiating two Pandas Data Frames as containers for csv files.
    # @skybiometry_fer as an easy-to-use container with the image name and the corresponding
    # emotion likelihoods as detected by Cloud Vision.
    facepp_fer = pd.DataFrame(columns=\
                              ['picture', 'happy', 'sad', 'surprise', 'fear',
                               'disgust', 'anger', 'contempt', 'neutral', 'confused'])
    # @skybiometry_json as a container for the raw data with the corresponding image name
    # as well as the time to execute the file in row index [-1]
    facepp_json = pd.DataFrame(columns=\
                              ['picture', 'response'])
    # initiating skybiometry client
    client = FacePPClient(FACEPP_KEY, FACEPP_JSON)
    # iterating over the specified folder with the images to be processed.
    for image in os.scandir(IMAGES_FOLDER):
        if image.name != '.DS_Store': # excludes generated file (by macOS)
            # running the  func to call the API and storing the
            # outputs in the predefined containers
            label_count = client.faces_detect(image.name, IMAGES_FOLDER)
            facepp_fer.loc[len(facepp_fer.index)] = label_count[0]
            facepp_json.loc[len(facepp_json.index)] = label_count[1]
            # tracking progress of processed images
            print(len(facepp_json.index))
            if len(facepp_json.index)%20 == 0:
                time.sleep(30)
            else:
                time.sleep(1.5)

        else:
            continue
    # add row with time used to execute script
    facepp_json.loc[len(facepp_json.index)] = ["time to execute: ", time.time() - start]
    # One-Hot-Encoding Output
    not_detected = facepp_fer.loc[facepp_fer['happy'] == 'no face detected']
    # drop 'no face detected' before one hot encoding
    facepp_fer['no face detected'] = facepp_fer['happy'] == 'no face detected'
    raw_fer = facepp_fer.replace('no face detected', 0)
    no_em_dect = raw_fer.iloc[:,1:].any(1).astype(int)
    raw_fer.insert(10, 'no emotions detected', no_em_dect)

    # ===================================================== ONE HOT ENCODING

    # Rescaling values to range from 0 to 1
    raw_fer.iloc[:,1:] = sklearn.preprocessing.MinMaxScaler().fit_transform(raw_fer.iloc[:,1:])
    # set all non-max values to 0 of each column (after transposing the df)
    raw_t = raw_fer.T.copy()
    raw_t.iloc[1:,:][raw_t.iloc[1:,:]!=raw_t.iloc[1:,:].max()] = 0
    # set the remaining non-zero values (maximum) to 1
    raw_t.iloc[1:,:][raw_t.iloc[1:,:] != 0] = 1
    encoded = raw_t.T

    # saving the outputs from the API call to .csv files
    not_detected.to_csv(FACEPP_NOTDETECTED)
    encoded.to_csv(FACEPP_OHE)
    facepp_fer.to_csv(FACEPP_FER)
    facepp_json.to_csv(FACEPP_JSON)
    print('\n05_FaceppRequest exe success')

if __name__ == "__main__":
    main()
