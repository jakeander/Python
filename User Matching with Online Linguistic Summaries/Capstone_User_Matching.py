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

#Loading the bokeh library
from bokeh.charts import Bar, show, defaults
from bokeh.io import gridplot, output_notebook

output_notebook()

defaults.width = 450
defaults.height = 450

#Creating a bokeh visual summary of the demographic variables
p1 = Bar(data, 'ORIENTATIONcode', legend = None, title = "Counts by Sexual Orientation", 
         xlabel = "Category of Sexual Orientation code", ylabel="")
p2 = Bar(data, 'RELIGIONcode', legend = None, title = "Counts by Religious Affiliation", 
         xlabel = "Category of Religious Affiliation code", ylabel="")
p3 = Bar(data, 'EDUCATIONcode', legend = None, title = "Counts by Highest Education Obtained", 
         xlabel = "Category of Education Attainment code", ylabel="")
p4 = Bar(data, 'GENDERCODED',legend = None, title = "Counts by Gender", 
         xlabel = "Category of Gender code", ylabel="")
show(gridplot([[p1,p2],[p3,p4]]))

#Visual summary of the linguistic summaries 
LIWC_sum_cat = ['Analytic', 'Clout', 'Authentic', 'Tone']

fig, axs = plt.subplots(1,4, sharex = True, sharey = True)
fig.set_size_inches(18, 6)

plot_num = -1

for feature in LIWC_sum_cat:
    plot_num += 1
    data['Analy_labels'] = data[feature] >= 50
    analytic_prob = data['Analy_labels'].value_counts()
    Analytic_Plot = analytic_prob.plot(kind='bar', ax = axs[plot_num], title = feature).set_xticklabels(["Above 50", "Below 50"])

data[LIWC_sum_cat].hist(figsize=(30,30), xlabelsize=20, ylabelsize=20)

#Loading the tsne algorithm and transform the data
from sklearn.manifold import TSNE 
model = TSNE(n_components=2, random_state=0)
t = model.fit_transform(data)

#Creating the color scheme for the tsne visuals
from sklearn import datasets
X1, color = datasets.samples_generator.make_s_curve(len(data), random_state=0)

#Showcasing the tsne reduced data
plt.figure(figsize=(20,20))
plt.scatter(t[:,0], t[:,1], c=color, cmap=plt.cm.Spectral)
plt.colorbar(ticks=range(10))
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.show()

