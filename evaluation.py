# evaluation.py
from sklearn.model_selection import cross_val_score
from sklearn.metrics import RocCurveDisplay, classification_report, confusion_matrix

def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    score = cross_val_score(model, X_test, Y_test, cv=10)
    
    print("Maximum Accuracy:", round(max(score) * 100, 2), "%")
    print("Average Accuracy:", round(score.mean() * 100, 2), "%")
    
    RocCurveDisplay.from_estimator(model, X_test, Y_test)
    
    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))
