import requests
import zipfile
from bs4 import BeautifulSoup
import io
import pandas as pd#creates nd arrays with much faster implementation than lists
import numpy as np#loading datasets,data manipulation etc
from sklearn.model_selection import train_test_split#
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import LabelEncoder,StandardScaler
import time
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
def load_data():
    page_url = 'https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones'
    page_response = requests.get(page_url)
    if page_response.status_code == 200:
        soup = BeautifulSoup(page_response.content, 'html.parser')
        download_link = soup.select_one('a[href$=".zip"]')['href']
        full_download_url = 'https://archive.ics.uci.edu' + download_link
        response = requests.get(full_download_url)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as outer_zip:
                inner_zip_name = 'UCI HAR Dataset.zip'
                with outer_zip.open(inner_zip_name) as inner_zip_file:
                    with zipfile.ZipFile(io.BytesIO(inner_zip_file.read())) as inner_zip:
                        with inner_zip.open('UCI HAR Dataset/train/X_train.txt') as myfile:
                            df = pd.read_csv(myfile, delim_whitespace=True, header=None)
                        with inner_zip.open('UCI HAR Dataset/train/y_train.txt') as myfile_y:
                            y = pd.read_csv(myfile_y, delim_whitespace=True, header=None)
    else:
        raise Exception("Failed to download or parse the dataset.")
    return df, y
df,y=load_data()
#eda
#print(y.head())
#print(df.describe())
print(y.isnull().sum())
#print(y.info())
#noneed of label encoding cause all are float types but still..
label_encoder=LabelEncoder()
categorical_cols = y.select_dtypes(include=['object']).columns.tolist()
print(categorical_cols)
#printing nothing so there is no categorical data in both df and y
#scaling 
scaler=StandardScaler()

scaler.fit(df)
df_scaled=scaler.transform(df)

#splitting in the training data into train and test data
X_train_full, X_test_full, y_train, y_test =train_test_split(df_scaled,y,test_size=0.2,random_state=7)
start_time = time.time()
classifier_pipeline_full = Pipeline([
    ('classifier', GaussianNB())
])
classifier_pipeline_full.fit(X_train_full, y_train)
y_pred_full = classifier_pipeline_full.predict(X_test_full)
end_time = time.time()
full_features_time = end_time - start_time
accuracy_full = accuracy_score(y_test, y_pred_full)
n_clusters = 50
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#now wt to do is pass this transposed into kmeans and then 
kmeans.fit(df_scaled.T)
selected_features_indices = [np.random.choice(np.where(kmeans.labels_ == i)[0]) for i in range(n_clusters)]

selected_features = df_scaled[:, selected_features_indices]

X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
    selected_features, y, test_size=0.2, random_state=7)

start_time2 = time.time()
classifier_pipeline_reduced = Pipeline([('classifier', GaussianNB())])
classifier_pipeline_reduced.fit(X_train_reduced, y_train_reduced)

y_pred_reduced = classifier_pipeline_reduced.predict(X_test_reduced)
end_time2 = time.time()
accuracy_reduced = accuracy_score(y_test_reduced, y_pred_reduced)
reduced_features_time = end_time2 - start_time2
print("Baseline Model (All Features):")
print("Accuracy:", accuracy_full)
print("Training Time:", full_features_time, "seconds")
print("Number of Features:", X_train_full.shape[1])

print("\nModel with Reduced Features (K-Means):")
print("Accuracy:", accuracy_reduced)
print("Training Time:", reduced_features_time, "seconds")
print("Number of Features:", n_clusters)
