# main.py
from data_loader import load_data
from eda import perform_eda
from preprocessing import preprocess_data
from visualize import visualize_data
from models import random_forest_model, logistic_regression_model
from evaluation import evaluate_model
from sklearn.model_selection import train_test_split

def main():
    # Load data
    data = load_data("indian_liver_patient.csv")
    
    # Perform EDA
    perform_eda(data)
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # Visualize data
    visualize_data(X, data)
    
    # Train Random Forest
    rf_model = random_forest_model(X_train, Y_train)
    evaluate_model(rf_model, X_test, Y_test)
    
    # Train Logistic Regression
    lr_model = logistic_regression_model(X_train, Y_train)
    evaluate_model(lr_model, X_test, Y_test)

if __name__ == "__main__":
    main()
