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

#Taking a quick look at the data after it was imported
data.info()

#Seeing the counts of values in some basick demographic variables
data[['ORIENTATIONcode','RELIGIONcode', 'EDUCATIONcode']].apply(pd.value_counts)

#Converting all the variables to a number value and viewing what it looks like
data = data.apply(pd.to_numeric, errors="coerce")
data.info()
data.head()
data.describe()

#Removing the variables with NaN's so it is a completely equal comparison between the benchmark model and the final model
data = data.dropna()
len(data)
print "The dataset from Sapio now has {} users for the analysis.".format(len(data))

#Filtering the dataset down to only include the linguistic summaries with more than 50 words
display(data.head())
data = data[data['WordCount'] > 50]
display(data.head())
print "The dataset from Sapio now has {} users for the analysis.".format(len(data))