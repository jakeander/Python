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

#Showing the DBSCAN result with the default epsilon and minimum samples configuration
#Bar chart
plt.figure(figsize=(15,10))
plt.bar(unique_counts[:,:1], unique_counts[:,1:])
plt.xlabel("Cluster Number")
plt.ylabel("Amount of Users in Cluster")
#plt.text(unique_counts[:,:1], unique_counts[:,1:], str(unique_counts[:,1:]))

for a,b in zip(unique_counts[:,:1], unique_counts[:,1:]):
    plt.text(a, b, str(b))
    
#Scatter plot
%matplotlib inline
plt.figure(figsize=(20,10))

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
         # Black used for noise.
        col = 'k'
 
    class_member_mask = (labels == k)
 
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
              markeredgecolor='k', markersize=14)
 
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
              markeredgecolor='k', markersize=6)
 
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

#Using a more optimal setting for the DBSCAN algorithm
X = StandardScaler().fit_transform(reduced_data)

db = DBSCAN(eps=0.095, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_

unique, counts = np.unique(labels, return_counts=True)

unique_counts = np.asarray((unique, counts)).T

print unique_counts[:,1:]

#Seeing the scatter plot result
%matplotlib inline
plt.figure(figsize=(20,10))

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
         # Black used for noise.
        col = 'k'
 
    class_member_mask = (labels == k)
 
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
              markeredgecolor='k', markersize=14)
 
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
              markeredgecolor='k', markersize=6)
 
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

#Scatter plot result without noise
%matplotlib inline
plt.figure(figsize=(20,10))

# Black removed and is used for noise instead.
unique_labels = set(labels[labels>=0])
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
         # Black used for noise.
        col = 'k'
 
    class_member_mask = (labels == k)
 
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
              markeredgecolor='k', markersize=14)
 
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
              markeredgecolor='k', markersize=6)
 
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

#Bar chart distribution of the users in the clusters
unique, counts = np.unique(labels[labels>=0], return_counts=True)

unique_counts = np.asarray((unique, counts)).T

plt.figure(figsize=(15,10))
plt.bar(unique_counts[:,:1], unique_counts[:,1:])
plt.xlabel("Cluster Number")
plt.ylabel("Amount of Users in Cluster")
plt.xticks(unique_counts[:,:1])

for a,b in zip(unique_counts[:,:1], unique_counts[:,1:]):
    plt.text(a, b, str(b))

plt.show()

#Creating a training and test set
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.20, random_state=15)

#Import modules for kmeans clustering
from sklearn import mixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

range_of_clusters = [2,3,4,5,6,7,8,9,10]

#Showing the silhouette score for the entire dataset

for n_clusters in range_of_clusters:
    clusterer = KMeans(n_clusters=n_clusters)
    clusterer.fit(X)

    preds = clusterer.predict(X)

    centers = clusterer.cluster_centers_

    score = silhouette_score(X, preds)
    print "For n_clusters = {}. The average silhouette_score is : {}".format(n_clusters, score)
    
#Finding the silhouette score for the training set
for n_clusters in range_of_clusters:
    clusterer = KMeans(n_clusters=n_clusters)
    clusterer.fit(X_train)

    preds = clusterer.predict(X_train)

    centers = clusterer.cluster_centers_

    score = silhouette_score(X_train, preds)
    print "For n_clusters = {}. The average silhouette_score is : {}".format(n_clusters, score)

#Finding the silhouette score for the test set
for n_clusters in range_of_clusters:
    clusterer = KMeans(n_clusters=n_clusters)
    clusterer.fit(X_test)

    preds = clusterer.predict(X_test)

    centers = clusterer.cluster_centers_

    score = silhouette_score(X_test, preds)
    print "For n_clusters = {}. The average silhouette_score is : {}".format(n_clusters, score)

#Using the optimal amount of clusters based on the silhouette scores
clusterer = KMeans(n_clusters=3)
clusterer.fit(X)
preds = clusterer.predict(X)
centers = clusterer.cluster_centers_

#Kmeans result
X1 = pd.DataFrame(X, columns = ['Dimension 1', 'Dimension 2'])
cluster_results(X1, preds, centers)

#Looking at the number of users in each of the clusters from the kmeans result
unique, counts = np.unique(clusterer.labels_, return_counts=True)

print np.asarray((unique, counts)).T
centers

#Looking at the true centers of the Kmeans cluster result
true_centers = pca.inverse_transform(centers)

segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(true_centers, columns = good_data.keys())
true_centers.index = segments
display(true_centers)

good_data.describe().loc[['mean']]
