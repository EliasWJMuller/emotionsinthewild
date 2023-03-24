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
import csv

from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from future.utils import iteritems
from face_client import multipart
from google.cloud import vision
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.optimize import minimize

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
