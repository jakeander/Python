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

#Creating the benchmark model with DBSCAN
#Seeing counts of the gender variable
data['GENDERCODED'].value_counts()

#Seeing the counts of the sexual orientation variable
data['ORIENTATIONcode'].value_counts()

# Import modules for DBSCAN
from sklearn import mixture
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

naive_vars = ['GENDERCODED', 'ORIENTATIONcode']
naive_data = data[naive_vars]

#Building the naive DBSCAN clustering
db = DBSCAN().fit(naive_data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_ 

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

X = StandardScaler().fit_transform(naive_data)

print('Estimated number of clusters: %d' % n_clusters_)

#DBSCAN clustering visual for the benchmark model
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

#Seeing the counts of the users in each clusters
unique, counts = np.unique(labels, return_counts=True)

print np.asarray((unique, counts)).T

#Feature Selection
#Using a decision tree regressor to see the relevance of the variable emotional tone
data_NoTone = data.drop(['Tone'], axis=1)
data_tone = data[['Tone']]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_NoTone, data_tone, test_size = 0.25, random_state = 5)

from sklearn import tree
regressor = tree.DecisionTreeRegressor(random_state = 5)
regressor.fit(X_train, y_train)

predict = regressor.predict(X_test)
score = regressor.score(X_test, y_test)
print(score)

#Applying Principal Component Analysis
sum_vars = ['Analytic', 'Clout', 'Authentic', 'Tone']
good_data = data[sum_vars]

sum_vars = ['Analytic', 'Clout', 'Authentic', 'Tone']
good_data = data[sum_vars]
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(good_data)

#Showing the PCA results
pca_results(good_data, pca)

#Using the PCA reduced data
pca = PCA(n_components = 2)
pca.fit(good_data)

reduced_data = pca.transform(good_data)
pca_samples = pca.transform(log_samples)
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

#Clustering the users using the reduced PCA datset with the DBSCAN algorithm
X = StandardScaler().fit_transform(reduced_data)

db = DBSCAN(eps=0.5).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_

unique, counts = np.unique(labels, return_counts=True)

unique_counts = np.asarray((unique, counts)).T

print unique_counts[:,1:]
