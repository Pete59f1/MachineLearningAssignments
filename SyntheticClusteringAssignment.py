import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Looking at the current dataset
# Looking at the data like this doesn't give us much info
data = pd.read_csv("s1-cb.txt", header=None)

# We will try to split the data into an X and Y array so we can use pyplot to illustrate the data
# First we make our data into a numpy array
dataArray = np.array(data)

# Now to split it into x and y arrays
x = []
y = []

# In this for loop we make every index in our array into a string
# We then split this string into two by splitting where space is
# We then add the pieces as integers to our x and y arrays
# The x gave some trouble so we use the join method to only add integers
for i in dataArray:
    string = str(i)
    splitter = string.split(' ')

    x.append(int(''.join(i for i in splitter[0] if i.isdigit())))
    y.append(int(splitter[1]))

# Looking at our illustrated data with pyplot
# Looking at the illustrated data its hard to say how many clusters there could be
plt.scatter(x, y)
plt.show()

# To begin with we want only two clusters: Flat clustering?
# We need to put our x and y arrays back into one so we can train our model with it
# There must be a better way to do this -_-
X = np.array([[x[0], y[0]], [x[1], y[1]], [x[2], y[2]], [x[3], y[3]], [x[4], y[4]], [x[5], y[5]], [x[6], y[6]],
              [x[7], y[7]], [x[8], y[8]], [x[9], y[9]], [x[10], y[10]], [x[11], y[11]], [x[12], y[12]], [x[13], y[13]],
              [x[14], y[14]]])

# Creating our model and tells it to cluster in two groups and then we train it
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Getting the centroids and labels. Gonna use these to illustrate
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# The colors we are gonna give our clusters
clusterColor = ['r.', 'g.']

# Using pyplot to illustrate how kmeans have clustered our data
# Also printing the coordinates and which cluster they belong to
for i in range(len(X)):
    print("Coordinate:", X[i], "Label:", labels[i])
    plt.plot(X[i][0], X[i][1], clusterColor[labels[i]], markersize=10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=100, linewidths=5, zorder=10)
plt.show()
