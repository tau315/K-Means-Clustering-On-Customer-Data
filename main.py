import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Extracting relevant data from CSV file

df = pd.read_csv("Mall_Customers.csv")
annual_income = df["Annual Income (k$)"].values
spending_score = df["Spending Score (1-100)"].values
data = np.column_stack((annual_income, spending_score))
#Plotting Data

plt.scatter(annual_income, spending_score, label = "Data")
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Data')
plt.legend()
#plt.show()

# Finding Ideal k
wcss_values = []
for k in range(1, 11):
    np.random.seed(0)
    initial = np.random.rand(2 * k)
    centroids = []
    for i in range(0, len(initial), 2):
        centroids.append([initial[i] * (max(annual_income) - min(annual_income)) + min(annual_income), initial[i + 1] * (max(spending_score) - min(spending_score)) + min(spending_score)])
    
    centroids = np.array(centroids)
    final_centroids = []
    for _ in range(100):
        #Calculating distances via broadcasting and assigning clusters for each point
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        closest_centroids = np.argmin(distances, axis=1)
        final_centroids = closest_centroids

        #Calculating new centroids via mean of points in each cluster
        new_centroids = []

        for i in range(k):
            points = data[closest_centroids == i]
            if len(points) > 0:
                new_centroids.append(np.mean(points, axis=0))
            else:
                new_centroids.append([initial[i] * (max(annual_income) - min(annual_income)) + min(annual_income), initial[i + 1] * (max(spending_score) - min(spending_score)) + min(spending_score)])

        new_centroids = np.array(new_centroids)

        #Checking for convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    wcss = 0
    for i in range(k):
        points = data[closest_centroids == i]
        wcss += np.sum(np.linalg.norm(points - centroids[i], axis=1) ** 2)
    wcss_values.append(wcss)

# Plotting Elbow Curve
plt.figure()
plt.plot(range(1, 11), wcss_values, label = "Elbow")
plt.xlabel('Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method: k = 5 is optimal')
plt.grid(True)
plt.show()


#Plotting k = 5
k = 5
np.random.seed(0)
initial = np.random.rand(2 * k)
centroids = []
for i in range(0, len(initial), 2):
    centroids.append([initial[i] * (max(annual_income) - min(annual_income)) + min(annual_income), initial[i + 1] * (max(spending_score) - min(spending_score)) + min(spending_score)])

centroids = np.array(centroids)
final_centroids = []
for _ in range(100):
    #Calculating distances via broadcasting and assigning clusters for each point
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    closest_centroids = np.argmin(distances, axis=1)
    final_centroids = closest_centroids

    #Calculating new centroids via mean of points in each cluster
    new_centroids = []

    for i in range(k):
        points = data[closest_centroids == i]
        if len(points) > 0:
            new_centroids.append(np.mean(points, axis=0))
        else:
            new_centroids.append([initial[i] * (max(annual_income) - min(annual_income)) + min(annual_income), initial[i + 1] * (max(spending_score) - min(spending_score)) + min(spending_score)])

    new_centroids = np.array(new_centroids)

    #Checking for convergence
    if np.allclose(centroids, new_centroids):
        break

    centroids = new_centroids

colors = ['red', 'blue', 'green', 'purple', 'orange']

plt.figure(figsize=(8, 6))

for i in range(k):
    points = data[final_centroids == i]
    plt.scatter(points[:, 0], points[:, 1], s=50, c=colors[i], label=f'Cluster {i+1}')

plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation (k = 5)')
plt.legend()
plt.grid(True)
plt.show()