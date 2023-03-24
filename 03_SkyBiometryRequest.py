# -*- coding: utf-8 -*-
"""03_SkyBiometryRequest.py

This .py file creates a Detect Client for the SkyBiometry API
and saves the output in .csv files when the API is called.

@FaceClient(): class creating the required form data and excecuting the request

@main(): excecutes the API Post for the AffectNet Sample and saves the
outputs as described below.

constants to be definded:
@WDIR: Path of working directory
@IMAGES_FOLDER: Path to folder with images to be processed
@SKYBIOMETRY_FER: Path to dedicated output file
@SKYBIOMETRY_JSON: Path to dedicated output file
@EX_SKYBIOMETRY: Path to dedicated output file
@SKY_BIOMETRY_KEY: SkyBiometry credentials
@SKY_BIOMETRY_SECRET: SkyBiometry credentials

@output: .csv files
    SkyBiometry_FER == emotions with corresponding confidence for each image
    SkyBiometry_JSON == raw json ouput for each image
    SkyBiometry_OHE == one hot encoded output

PREREQUISITES:
    1. SkyBiometry Account SetUp
    2. Client Authentication (api_key and api_secret)
    find details --> https://skybiometry.com/documentation/
    3. Execute the following instructions for the multipart.py module

IMPORNANT NOTE:
<from face_client import multipart> imports a deprecated module.
After import the following modification is required in the multipart.py module for Python 3.X:
    line 75 add:
        if isinstance(self._body, bytes):
            self._body = self._body.decode('latin')
This is required to convert the byte string of the image body (MIME: image/jpeg) to a
compatible format. If no modification is made line 141 (<return content_type, Part.CRLF.join(all))
will trigger the following error as .join(all) has been newly defined:
    TypeError: sequence item {integer}: expected str instance, bytes found
"""
# importing the required modules
import os.path
import time
import json
import requests
import pandas as pd
import numpy as np
from future.utils import iteritems
from face_client import multipart # modification required
from sklearn.preprocessing import OneHotEncoder


#constants to be defined
WDIR = 'PATH_TO_WDIR'
IMAGES_FOLDER = 'PATH_TO_IMAGES'
SKYBIOMETRY_FER = 'PATH_TO_OUTPUT_FER'
SKYBIOMETRY_JSON = 'PATH_TO_OUTPUT_JSON'
SKYBIOMETRY_NOTDETECTED = 'PATH_TO_SAFE_UNDETECTED'
SKYBIOMETRY_OHE = 'PATH_TO_OUTPUT_OHE'
SKYBIOMETRY_KEY = 'API_KEY'
SKYBIOMETRY_SECRET = 'API_SECRET'

# DO NOT MODIFY
API_HOST = 'api.skybiometry.com/fc' # base url
USE_SSL = True # http or https

class FaceClient(object):
    """FaceClient(object)

    Modified extract of the SkyBiometry Library (Maurus, T.)
    Original Python Source Code Library:
    Maraus, T. (NA). python-face-client.
    Retrieved from https://github.com/SkyBiometry/python-face-client

    Name: SkyBiometry Face Detection and Recognition API Python client library
    Description: SkyBiometry Face Detection and Recognition REST API Python client library.

    For more information about the API and the return values,
    visit the official documentation at http://www.skybiometry.com/Documentation

    Author: Tomaž Muraus (http://www.tomaz.me)
    License: BSD

    ===========================================================================

    """
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.format = 'json'

    def faces_detect(self, name, file=None, aggressive=False):
        """faces_detect(self, name, file=None, aggressive=False)
        
        Returns tags for detected faces in one or more photos, with geometric
        information of the tag, eyes, nose and mouth, as well as the gender,
        glasses, and smiling attributes.
        http://www.skybiometry.com/Documentation#faces/detect
        
        """
        # specify parameters for the API request
        # @params: attributes:['all, 'none', {list of attributes}]
        data = {'attributes': 'all', 'force_reprocess_image': 'true'}
        files = []
        # generate actual path to file in container
        files.append(file+name)

        # aggressive forces more accurate detection
        if aggressive:
            data['detector'] = 'aggressive'

        # call self.send_request to faces/detect endpoint
        response = self.send_request('faces/detect', data, files)
        return response

    def send_request(self, method=None, parameters=None, files=None):
        """send_request(self, method=None, parameters=None, files=None)
        
        Use send_request to call the SkyBiometry API for applicable images.

        @method: define endpoint
        @parameters: (Key, Secret, Detector)
        @files: path to file

        @return: API output defined in generate_response(self, image_name, json_response

        """
        # what kind of protocol is being used (defined as constant)
        protocol = 'https://' if USE_SSL else 'http://'
        # generate callable url
        url = '%s%s/%s.%s' % (protocol, API_HOST, method, self.format)
        # generate container for required credentials
        data = {'api_key': self.api_key, 'api_secret': self.api_secret}

        if parameters:
            data.update(parameters) # {'attributes': 'all', 'force_reprocess_image': 'true'}

        # Local file is provided, use multi-part form
        if files:
            # create Multipart instance
            form = multipart.Multipart()
            # Parameters added to form data (Key, Secret, Detector, Attributes & Reprocess)
            for key, value in  iteritems (data):
                form.field(key, value)
            # iterate over files provided to attach the respecive form bodies
            for file in files:
                name = os.path.basename(file)
                file = open(file, 'rb')
                close_file = True
                try:
                    form.file(name, name, file.read())
                finally:
                    if close_file:
                        file.close()
            # get the multipart form data
            (content_type, post_data) = form.get()
            # specify the content type
            headers = {'Content-Type': content_type}

        # post the API request with the defined parameters (form body: image/jpeg)
        req = requests.post(url, headers=headers, data=post_data)
        # conversion from html to text
        response = req.text
        # convert to dictionary
        response_data = json.loads(response)
        # convert response
        conv_response = self.generate_response(name, response_data)
        
        return conv_response
    
    def generate_response(self, image_name, json_response):
        """generate_response(self, image_name, json_response)

        Use send_request to call the SkyBiometry API for applicable images.

        @image_name: name of input image
        @json_response: API response to be processed

        @return: Output depending on API response

        """
        face = json_response['photos'][0]['tags']
        if len(face) == 0:
            return ([image_name, 'no face detected', 'no face detected', 'no face detected', 'no face detected', 
                     'no face detected','no face detected', 'no face detected', 'no face detected'],
                        [image_name, json_response])
        try: 
            emotions = face[0]['attributes']
            return ([image_name, emotions['mood'], emotions['neutral_mood'], emotions['anger'], emotions['disgust'],
                          emotions['fear'], emotions['happiness'], emotions['sadness'], emotions['surprise']],
                        [image_name, json_response])
        except: 
            return ([image_name, 'no emotion detected', 'no emotion detected', 'no emotion detected', 'no emotion detected', 
                     'no emotion detected','no emotion detected', 'no emotion detected', 'no emotion detected'],
                        [image_name, json_response])
      

def main():
    """main()

    Executed when run from the command line, defining the operational functionality
    of the script while excluding potentially ressource heavy functions.

    """
    print('03_SkyBiometry exe')
    start = time.time()
    # specifying the working directory
    os.chdir(WDIR)
    # initiating two Pandas Data Frames as containers for csv files.
    # @skybiometry_fer as an easy-to-use container with the image name and the corresponding
    # emotion likelihoods as detected by Cloud Vision.
    skybiometry_fer = pd.DataFrame(columns=\
                              ['picture', 'mood', 'neutral_mood', 'anger', 'disgust',
                               'fear', 'happiness', 'sadness', 'surprise'])
    # @skybiometry_json as a container for the raw data with the corresponding image name
    # as well as the time to excecute the file in row index [-1]
    skybiometry_json = pd.DataFrame(columns=\
                              ['picture', 'response'])
    # initiating skybiometry client
    client = FaceClient(SKYBIOMETRY_KEY, SKYBIOMETRY_SECRET)
    # iterating over the specified folder with the images to be processed.
    for image in os.scandir(IMAGES_FOLDER):
        if image.name != '.DS_Store': # excludes generated file (by macOS)
            # running the  func to call the API and storing the
            # outputs in the predefined containers
            label_count = client.faces_detect(image.name, file=IMAGES_FOLDER, aggressive=True)
            skybiometry_fer.loc[len(skybiometry_fer.index)] = label_count[0]
            skybiometry_json.loc[len(skybiometry_json.index)] = label_count[1]
            # tracking progress of processed images
            print(len(skybiometry_json.index))
            # pause loop to not exceed API limits
            time.sleep(40)
        else:
            continue
    # add row with time used to excute script
    skybiometry_json.loc[len(skybiometry_json.index)] = ["time to excecute: ", time.time() - start]
    
    # filter out results where SkyBiometry Vision did not detect a face
    not_detected = skybiometry_fer.loc[skybiometry_fer['mood'] == 'no face detected']
    # select the column with the detected mood and extract the keyword
    raw_mood = (skybiometry_fer.iloc[:,1])
    # convert 'no emotions deteted' to same format as the rest of the df
    raw_mood = raw_mood.replace("no emotions detected", "{'value': 'no emotions detected'")
    # same replacement value can be used, as result for OHE remains the same 
    raw_mood = raw_mood.replace("no face detected", "{'value': 'no face detected'")
    
    # extract mood keyword only
    # Note: 'no emotions deteted' currently a category. Is be dropped in the final df
    raw_mood = raw_mood.astype('string')
    raw_mood = raw_mood.str.split("'").str[3]
    
    # ===================================================== ENCODING
    
    # One Hot Encoding of the sample labels
    
    extract_np = np.asarray(raw_mood)
    integer_encoded = extract_np.reshape(len(extract_np), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    encoded = pd.DataFrame(columns=['picture', 'angry', 'disgusted', 'happy', 'neutral', 'no emotions detected',
                                    'no face detected', 'sad', 'scared', 'surprised'])
    encoded['picture'] = skybiometry_fer.iloc[:,0]
    encoded.iloc[:,1:] = onehot_encoded
    
    encoded = encoded.rename(columns={'surprised': 'surprise', 'scared': 'fear', 'disgusted': 'disgust',
                                      'angry': 'anger',})
    
    encoded = encoded.reindex(columns=['picture', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger',
                                       'contempt', 'neutral', 'confused', 'no emotions detected', 'no face detected'],
                    fill_value = 0)
    

    # saving the outputs fromn the API call to .csv files
    skybiometry_fer.to_csv(SKYBIOMETRY_FER)
    skybiometry_json.to_csv(SKYBIOMETRY_JSON)
    not_detected.to_csv(SKYBIOMETRY_NOTDETECTED)
    encoded.to_csv(SKYBIOMETRY_OHE)

    print('\n03_SkyBiometry exe success')

# basic guard to protect from accidental excecution of the script
if __name__ == "__main__":
    main()
