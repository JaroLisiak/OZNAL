from builtins import print

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.utils.testing import (assert_raises, assert_greater, assert_equal, assert_false)
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


rawData = pd.read_csv("data.csv")

window_size = 300
predicted_array = []
real_array = []
error1 = 0
error2 = 0.0
count = 0
try:

    prepdata2 = rawData[(rawData['hour'] > 6) & (rawData['hour'] < 19) & (rawData['plantPower'] == 0)].index #(rawData['hour'] > 6) & (rawData['hour'] < 19) &
    # Hodiny jednotlivo

    prepdata = rawData.drop(prepdata2).reset_index()
    prepdata=rawData.drop(columns=['date','time','id','recordInDay','plantIrradiance','plantTemperature','hour','weatherTemperature'])



    for index, row in prepdata.iterrows():
        if index%100 == 0:
            print(index," / ",  len(prepdata))

        if index+window_size+1 >= len(prepdata):
            break
        X_train = prepdata.loc[index:index+window_size].drop(columns=['plantPower']) # na tomto sa uci
        X_test = prepdata.loc[[index+window_size+1]].drop(columns=['plantPower']) # na zaklade tohto chcem predpovedat
        Y_train = prepdata.loc[index:index+window_size]['plantPower'] # toto sa snazi trafit pocas ucenia
        Y_test =  prepdata.loc[[index+window_size+1]]['plantPower'] # toto chcem predpovedou ziskat

        #Linerna regressia
        reg = linear_model.LinearRegression()
        reg.fit(X_train,Y_train)
        predicted = reg.predict(X_test)

        # MLP - multi-layer perceptron
        # mlp = MLPRegressor(solver='lbfgs', activation='identity', hidden_layer_sizes=(10,), max_iter=1000)
        # mlp.fit(X_train, Y_train)
        # predicted = mlp.predict(X_test)

        # add result into global variables
        predicted_array.insert(index, predicted)
        real_array.insert(index, Y_test[index+window_size+1])
        count += 1
        error1 += mean_squared_error(Y_test, predicted)


    
    error2 += r2_score(real_array, predicted_array)
    print("Mean squared error: ", error1 / count)
    print("R2 score: ", error2)
    print("MAPE: ", mean_absolute_percentage_error(real_array, predicted_array))

    print("done")







    # # Average day + linear regression
    # dataa = rawData.groupby(['dayID']).mean()
    # prepdata = dataa.drop(columns=['id','recordInDay','plantIrradiance','plantTemperature','hour'])
    # X_train = prepdata.drop(columns=['plantPower'])[:-31]
    # X_test = prepdata.drop(columns=['plantPower'])[-31:]
    # Y_train = prepdata['plantPower'][:-31]
    # Y_test = prepdata['plantPower'][-31:]
    #
    # #Linerna regressia
    # reg = linear_model.LinearRegression()
    # reg.fit(X_train,Y_train)
    # predicted = reg.predict(X_test)
    #
    # plt.title("Linearna predikcia priemernych dni")
    # plt.ylabel("Priemerny vykon elektrarne")
    # plt.xlabel("Predpovedany den")
    #
    # Y_test = Y_test.reset_index(drop=True)
    # plt.plot(Y_test)
    # plt.plot(predicted)
    # plt.plot(predicted,marker='o',markersize=4,color='orange')
    # plt.legend(['Original', 'Predicted'], loc=9 )
    # plt.show()
    #
    # # The coefficients
    # print('Coefficients: \n', reg.coef_)
    # # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(Y_test, predicted))
    # # Explained variance score: 1 is perfect prediction
    # print('Variance score: %.2f' % r2_score(Y_test, predicted))



    # # Logistic regression + hours
    # # zistili sme ze sa pouziva pre predpoved clasifikacie, teda predpoveda len 0 a 1 v kategorii
    # prepdata = rawData.drop(columns=['id','recordInDay','plantIrradiance','plantTemperature','hour','date','time','dayID'])
    # X_train = prepdata.drop(columns=['plantPower'])[:-589]
    # X_test = prepdata.drop(columns=['plantPower'])[-589:]
    # Y_train = prepdata['plantPower'][:-589]
    # Y_test = prepdata['plantPower'][-589:]
    # logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    # # Create an instance of Logistic Regression Classifier and fit the data.
    # logreg.fit(X_train, Y_train)
    # Z = logreg.predict(X_test)
    # print("Accuracy of logistic regression classifier on test set:" + Z)



    # # Mlp + hours
    # prepdata = rawData.drop(columns=['id','recordInDay','plantIrradiance','plantTemperature','hour','date','time','dayID'])
    #
    # #for index in prepdata:
    # X_train = prepdata.drop(columns=['plantPower'])[:-6953]
    # X_test = prepdata.drop(columns=['plantPower'])[-6953:]
    # Y_train = prepdata['plantPower'][:-6953]
    # Y_test = prepdata['plantPower'][-6953:]
    # mlp = MLPRegressor()
    # mlp.fit(X_train, Y_train)
    # y_predict = mlp.predict(X_test)
    #
    # print("Mean squared error: %.2f" % mean_squared_error(Y_test, y_predict))
    # # Explained variance score: 1 is perfect prediction
    # print('Variance score: %.2f' % r2_score(Y_test, y_predict))
    # #assert_greater(mlp.score(X_train, Y_train), 0.9)
    # plt.plot(Y_test.reset_index())
    # plt.plot(y_predict)
    # plt.show()


    #MPL + average day
    # dataa = rawData.groupby(['dayID']).mean()
    # prepdata = dataa.drop(columns=['id','recordInDay','plantIrradiance','plantTemperature','hour'])
    # X_train = prepdata.drop(columns=['plantPower'])[:-31]
    # X_test = prepdata.drop(columns=['plantPower'])[-31:]
    # Y_train = prepdata['plantPower'][:-31]
    # Y_test = prepdata['plantPower'][-31:]
    # mlp = MLPRegressor(solver= 'lbfgs',activation='identity',hidden_layer_sizes=(10,),max_iter=1000)
    # mlp.fit(X_train, Y_train)
    # y_predict = mlp.predict(X_test)
    # print(mlp.intercepts_);
    # print("Pocet iteracii solvera: %.2f" % mlp.n_iter_)
    # print("Mean squared error: %.2f" % mean_squared_error(Y_test, y_predict))
    # # Explained variance score: 1 is perfect prediction
    # print('Variance score: %.2f' % r2_score(Y_test, y_predict))
    # #assert_greater(mlp.score(X_train, Y_train), 0.9)
    # plt.plot(Y_test.reset_index(drop=True))
    # plt.plot(y_predict)
    # plt.legend(['Original', 'Predicted'], loc=9)
    # plt.show()







    # pokus o gaussovu regresiu
    # kernel = DotProduct() + WhiteKernel()
    # gpr = GaussianProcessRegressor(kernel=kernel, random_state = 100)
    # gpr.fit(X_train, Y_train)
    #
    # print(gpr.score(X_train, Y_train))
    # print("done")

    # knn = KNeighborsRegressor()
    # knn.fit(train[['dayID', 'hour', 'weatherCloudCover', 'weatherDewPoint', 'weatherHumidity', 'weatherPressure', 'weatherTemperature', 'weatherWindBearing', 'weatherWindSpeed']], train[['plantPower']])
    # predicted = knn.predict(test[['dayID', 'hour', 'weatherCloudCover', 'weatherDewPoint', 'weatherHumidity',  'weatherPressure', 'weatherTemperature', 'weatherWindBearing', 'weatherWindSpeed']])


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





