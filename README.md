# Liver Disease Prediction Project
This project involves building a machine learning model to predict liver disease using a dataset of patient medical records. The dataset is preprocessed, and two machine learning models — Random Forest and Logistic Regression — are trained to classify whether a patient has liver disease.

## Project Structure
The project is modularized into different components for ease of use and maintainability:

### data_preprocessing.py: 
Handles the data loading, cleaning, preprocessing, and scaling of features.
### exploratory_analysis.py: 
Performs exploratory data analysis (EDA), including visualization of distributions, correlations, and patterns in the dataset.
### model_training.py: 
Contains code for training the machine learning models (Random Forest and Logistic Regression) with hyperparameter tuning.
### evaluation.py: 
Evaluates the performance of the trained models using metrics like accuracy, confusion matrix, and ROC curve.
### main.py: 
The entry point for running the entire pipeline, from data preprocessing to model evaluation.

## How It Works
### Data Preprocessing:
The dataset is loaded, cleaned by removing duplicates, and missing values are handled.
Numerical features are scaled using StandardScaler.
Categorical features are encoded using one-hot encoding.

### Exploratory Data Analysis:
Visualizes the distribution of features and the correlation between them.
Displays histograms and correlation heatmaps to better understand the dataset.

### Model Training:
Two models are trained: Random Forest and Logistic Regression.
Hyperparameter tuning is performed using GridSearchCV to find the best parameters for each model.

### Model Evaluation:
The models are evaluated using 10-fold cross-validation to get an estimate of their performance.
Performance metrics such as accuracy, confusion matrix, and ROC curves are generated to assess the model's effectiveness.

## Dataset
The dataset used is the Indian Liver Patient Dataset (ILPD), which contains medical data of patients with liver disease. The target variable Dataset indicates whether a patient has liver disease (1) or not (0).