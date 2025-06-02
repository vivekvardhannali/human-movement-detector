import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import time
from sklearn.model_selection import train_test_split

# Assuming df_scaled and y are already defined

cluster_range = [10, 50, 100, 200, 300, 400, 500]
accuracy_list = []
time_list = []

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(df_scaled.T)
    
    selected_features_indices = [np.random.choice(np.where(kmeans.labels_ == i)[0]) for i in range(n_clusters)]
    selected_features = df_scaled[:, selected_features_indices]
    
    X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.2, random_state=7)
    
    classifier_pipeline = Pipeline([
        ('classifier', GaussianNB())
    ])
    
    start_time = time.time()
    classifier_pipeline.fit(X_train, y_train)
    y_pred = classifier_pipeline.predict(X_test)
    end_time = time.time()
    
    accuracy = accuracy_score(y_test, y_pred)
    train_time = end_time - start_time
    
    accuracy_list.append(accuracy)
    time_list.append(train_time)

# Plot Accuracy vs Number of Clusters
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(cluster_range, accuracy_list, marker='o')
plt.title('Accuracy vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Accuracy')
plt.grid(True)

# Plot Training Time vs Number of Clusters
plt.subplot(1,2,2)
plt.plot(cluster_range, time_list, marker='o', color='orange')
plt.title('Training Time vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Training Time (seconds)')
plt.grid(True)

plt.tight_layout()
plt.show()
