import numpy as np

'''*********Basic cluster***************'''
# Correctly define the data array
data = np.array([
    [30, 50000],
    [40, 70000],
    [50, 55000],
    [70, 70000]
])

# Define the new data point
new_data = np.array([30, 100000])

# Reshape new_data to match the dimensions of data for broadcasting
new_data = new_data.reshape(1, -1)

# Calculate distances to centroids
distances_to_centroids = np.linalg.norm(data - new_data, axis=1)

# Find index of the closest centroid
closest_cluster_index = np.argmin(distances_to_centroids)

print('New customer assigned to cluster:', closest_cluster_index + 1)

print('''\n ********** Next Method ********* \n''')

from sklearn.cluster import KMeans

np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.0f}'.format})

data = np.array([
    [30, 60000],
    [35, 70000],
    [40, 80000],
    [45, 90000],
    [50, 100000],
    [55, 110000],
    [25, 50000],
    [60, 1200000],
    [90, 19000000]
])

new_data = np.array([92, 1980000])

# create a KMeans instance with 3 clusters
Kmeans = KMeans(n_clusters=3, random_state=0)
Kmeans.fit(data)

print(f'Cluster Centers \n {Kmeans.cluster_centers_}')

Labels = Kmeans.predict(data)
print('Cluster Labels for existing data points: ', Labels)

# Predict the cluster for the new customer
new_data_cluster = Kmeans.predict([new_data])

print('New data assigned to cluster', new_data_cluster[0])

print('\n **************** Next Method *********************** \n')


data = np.array([
    [30, 60000],
    [35, 70000],
    [40, 80000],
    [45, 90000],
    [50, 100000],
    [55, 110000],
    [25, 50000],
    [60, 1200000],
    [90, 19000000]
])

new_data = np.array([20, 50000])

# Generating a random state

random_state = np.random.randint(0, 100)

print('Random state values: ', random_state)

# create a KMeans instance with 3 clusters
Kmeans = KMeans(n_clusters=4, random_state=random_state)
Kmeans.fit(data)

print(f'Cluster Centers \n {Kmeans.cluster_centers_}')

Labels = Kmeans.predict(data)
print('Cluster Labels for existing data points: ', Labels)

# Predict the cluster for the new customer
new_data_cluster = Kmeans.predict([new_data])

print('Predict', new_data_cluster)

# Mapping of cluster Labels to names

cluster_names = {
    0: 'Young Low-Income Group',
    1: 'Middle-Aged Medium-Income Group',
    2: 'Senior High-Income Group',
    3: 'Other Groups'
}

# Get the names of the cluster for the new data

new_data_cluster_name = cluster_names[new_data_cluster[0]]

'''Since new_data_cluster is a numpy array, you need to access the first (and only) element to 
get the actual cluster label. This is done using new_data_cluster[0]'''

print('New data assigned to cluster:', new_data_cluster_name)




















