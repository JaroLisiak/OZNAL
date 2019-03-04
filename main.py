import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

rawData = pd.read_csv("data.csv")

try:
    #corr = rawData.corr()
    #correlation matrix
    #sns.heatmap(corr,annot=True,center=0,robust=True,cmap="Blues")

    #outliers
    #graf = sns.boxplot(x=rawData['plantPower'])
    #graf = sns.swarmplot(x=rawData['plantIrradiance'])

    # vykon / ozarovanie plot
    # dataa = rawData.groupby(['dayID']).sum()
    # plt.subplot(211)
    # plt.title("Výkon elektrárne")
    # plt.xlabel("Deň v roku")
    # plt.ylabel("Výkon")
    # plt.plot(dataa['plantPower'])
    # plt.subplot(212)
    # plt.title("Ožarovanie článkov")
    # plt.xlabel("Deň v roku")
    # plt.ylabel("Hodnota ožarovania")
    # plt.plot(dataa['plantIrradiance'])
    # plt.subplots_adjust(wspace=50,hspace=0.1)


    #plt.plot(data=dataa)
    # plt.subplot(212)
    # plt.plot(rawData['id'], rawData['plantIrradiance'], '-')
    #plt.xlabel('ID záznamu')
    #plt.ylabel('plantPower')
    #print(dataa.head)
    #sns.lineplot(data=dataa, x='dayID',y='plantPower')

    #sns.pairplot(rawData)
    plt.show()
except Exception as e:
    print(e)



