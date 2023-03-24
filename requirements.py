import os
import os.path
import random
import shutil
import time
import boto3
import json
import requests
import urllib
import base64

from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from future.utils import iteritems
from face_client import multipart
from google.cloud import vision

import numpy as np
import pandas as pd



