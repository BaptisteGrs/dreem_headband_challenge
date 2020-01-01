# Baptiste GROSS
# December 26th 2019
# Dreem Sleep Stages Classification Challenge 

# basic libraries

import h5py
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as sg
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn import metrics 
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# more exotic libraries
from entropy import entropy
