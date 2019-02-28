import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

rawData = pd.read_csv("data.csv")

try:
    corr = rawData.corr()

    sns.heatmap(corr,annot=True,center=0,robust=True,cmap="Blues")

    plt.show()
except Exception as e:
    print(e)



