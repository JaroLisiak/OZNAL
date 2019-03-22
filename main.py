import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

rawData = pd.read_csv("data.csv")

try:
    prepdata = rawData.drop(columns=['time','date','id','recordInDay','plantIrradiance','plantTemperature','weatherPressure','weatherWindSpeed','dayID','hour','recordInDay'])

    X_train = prepdata.drop(columns=['plantPower'])[:-589]
    X_test = prepdata.drop(columns=['plantPower'])[-589:]

    Y_train = prepdata['plantPower'][:-589]
    Y_test = prepdata['plantPower'][-589:]

    reg = linear_model.LinearRegression()
    reg.fit(X_train,Y_train)
    predicted = reg.predict(X_test)

    # The coefficients
    print('Coefficients: \n', reg.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(Y_test, predicted))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(Y_test, predicted))


    # knn = KNeighborsRegressor()
    # knn.fit(train[['dayID', 'hour', 'weatherCloudCover', 'weatherDewPoint', 'weatherHumidity', 'weatherPressure', 'weatherTemperature', 'weatherWindBearing', 'weatherWindSpeed']], train[['plantPower']])
    # predicted = knn.predict(test[['dayID', 'hour', 'weatherCloudCover', 'weatherDewPoint', 'weatherHumidity',  'weatherPressure', 'weatherTemperature', 'weatherWindBearing', 'weatherWindSpeed']])

    print("done")
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
    # plt.show()
except Exception as e:
    print(e)



