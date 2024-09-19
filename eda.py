# eda.py
import matplotlib.pyplot as plt
import pandas as pd

def perform_eda(data):
    print(data.info())
    
    class_count = pd.value_counts(data['Dataset'], sort=True).sort_index()
    class_count.plot(kind='bar')
    plt.title("Liver disease classes")
    plt.xlabel("Dataset")
    plt.ylabel("Frequency")
    plt.show()
