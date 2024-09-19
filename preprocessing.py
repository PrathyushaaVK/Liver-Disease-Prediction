# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Map Dataset column values
    data['Dataset'] = data['Dataset'].map({2: 0, 1: 1})
    
    # Remove duplicates
    data = data[~data.duplicated(keep='first')]
    
    # Fill missing values
    data['Albumin_and_Globulin_Ratio'].fillna(value=0, inplace=True)
    
    # Scale numerical features
    num_features = data.drop(['Gender', 'Dataset'], axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(num_features)
    
    # One-hot encoding for categorical features
    data_exp = pd.get_dummies(data)
    
    return data_exp, data['Dataset']
