## This script was for my capstone that was completed for my Machine Learning Nanodegree from Udacity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display 

# Pretty display for notebooks
%matplotlib inline

# Load the sapio user dataset
try:
    data = pd.read_csv("Sapio-Data-Anonymized-v2.csv")
    print "The dataset from Sapio has {} users with {} features each.".format(*data.iloc[:,1:].shape)
except:
    print "Is the data missing?"
