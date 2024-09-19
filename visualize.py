# visualize.py
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(X, data):
    X.hist(figsize=(15, 10))
    plt.ylabel("Frequency")
    plt.show()

    sns.catplot(x="Gender", col="Dataset", data=data, kind="count", height=5, aspect=.8)
    sns.catplot(x="Age", col="Dataset", data=data, kind="count", height=20, aspect=1.0)
    plt.show()
