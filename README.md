
# Human Activity Recognition (HAR) - Failure Rate Predictor  

This project uses smartphone sensor data to predict human activities using machine learning. The dataset was scraped from the UCI Machine Learning Repository using BeautifulSoup, and various preprocessing and modeling techniques were applied for classification.  

##  Overview

-> Objective: Classify different physical activities (e.g., walking, standing, laying) based on sensor signals.  
-> Dataset: UCI HAR Dataset (collected via smartphones worn on the waist).  
-> ML Goal: Predict activity type using statistical features extracted from accelerometer and gyroscope signals.  

##  Dataset

-> Source: Scraped from the UCI repository directly using Python's `requests` and `BeautifulSoup`.   
-> Features: 561 float-type features representing sensor readings.  
-> Target Variable: Activity labels encoded as integers (1 to 6).  
 
## 🛠 Methods and Techniques 

-> Web Scraping: Automated dataset download and extraction from nested ZIP files.  
-> Preprocessing:  
   - Used `StandardScaler` to normalize sensor data.  
   - Checked and confirmed no missing or categorical data.  
-> Dimensionality Reduction:  
   - Applied K-Means clustering to group similar features.  
   - Selected representative features from each cluster to reduce total feature count to 50.  
-> Classification:  
   - Trained a Gaussian Naive Bayes classifier on both the full and reduced datasets.  
   - Evaluated performance using accuracy score and timing metrics.  

##  Results  

-> Baseline Model (561 Features):  
   - Accuracy: 71.22%  
   - Training Time: 0.097 sec  

-> Reduced Model (50 Features via K-Means):  
   - Accuracy: 84.36%  
   - Training Time: 0.0105 sec  

##  Key Highlights  

-> Automated end-to-end pipeline: From data fetching to model training and evaluation.  
-> Feature reduction using K-Means significantly improved computational efficiency.  
-> Demonstrated effectiveness of Naive Bayes in high-dimensional sensor data.  
-> Model adaptable for fitness apps, fall detection systems, and healthcare monitoring.  
  
##  Tools & Libraries  

-> Python  
-> pandas, numpy, scikit-learn  
-> BeautifulSoup (for scraping)  
-> Google Colab environment

