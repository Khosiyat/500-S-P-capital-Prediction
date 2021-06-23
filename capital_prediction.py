from collections import Counter
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style

dataFrame=pd.read_csv("compiledStockTickers.csv")
#dataFrame.head(3)

def dataProcessing_step_toCreateLabels(stockTicker):
    day_num = 7
    stockTickers = dataFrame.columns.values.tolist()
    dataFrame.fillna(0, inplace=True)

    for day in range(1,day_num+1):
        dataFrame['{}_{}d'.format(stockTicker,day)] = (dataFrame[stockTicker].shift(-day) - dataFrame[stockTicker]) / dataFrame[stockTicker]

    dataFrame.fillna(0, inplace=True)
    return stockTickers, dataFrame

dataProcessing_step_toCreateLabels('ADBE')



def preconditionForDecisionMaking(*columns):
    columns=[column for column in columns]
    preconditionForDecisionMaking=0.02
    for column in columns:
        if column>preconditionForDecisionMaking:
            return 1#buy
        if column<-preconditionForDecisionMaking:
            return -1#sell
    return 0#hold



def featuresetsExtraction_step(stockTicker):
    stockTickers, dataFrame = dataProcessing_step_toCreateLabels(stockTicker)

    dataFrame['{}_target'.format(stockTicker)] = list(map( preconditionForDecisionMaking,
                                               dataFrame['{}_1d'.format(stockTicker)],
                                               dataFrame['{}_2d'.format(stockTicker)],
                                               dataFrame['{}_3d'.format(stockTicker)],
                                               dataFrame['{}_4d'.format(stockTicker)],
                                               dataFrame['{}_5d'.format(stockTicker)],
                                               dataFrame['{}_6d'.format(stockTicker)],
                                               dataFrame['{}_7d'.format(stockTicker)] ))
    values_stockTickerList = dataFrame['{}_target'.format(stockTicker)].values.tolist()
    strValues_stockTickerList = [str(stockTicker) for stockTicker in values_stockTickerList]
    print("This is Variation's:",Counter(strValues_stockTickerList), '\n')

    infinity= np.inf
    dataFrame.fillna(0, inplace=True)
    dataFrame = dataFrame.replace([infinity, - infinity], np.nan)
    dataFrame.dropna(inplace=True)
    dataFrame.replace(r'^\s*$', np.nan, regex=True)
#     dataFrame["Date"] = [float(str(i).replace(",", "")) for i in dataFrame["Date"]]
    day=dataFrame['Date']
    pd.to_datetime(day)

#     percent_change = dataFrame[[stockTicker for stockTicker in stockTickers]].pct_change()
    dataFrame_value = dataFrame.replace([infinity, -infinity], 0)
    dataFrame_value.fillna(0, inplace=True)

    X = dataFrame_value.values
    y = dataFrame['{}_target'.format(stockTicker)].values

    return X,y,dataFrame

featuresetsExtraction_step('WYNN')


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
for column_name in dataFrame.columns:
    if dataFrame[column_name].dtype == object:
        dataFrame[column_name] = le.fit_transform(dataFrame[column_name])
    else:
        pass



from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def machineLearnig_step(stockTicker):
    X, y, dataFrame = featuresetsExtraction_step(stockTicker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    linearSVC=svm.LinearSVC()
    knn=neighbors.KNeighborsClassifier()
    randomForest=RandomForestClassifier()

    voting_classifier = VotingClassifier([
                                          ('knn', knn),
                                          ('randomForest', randomForest)])

    

    
    voting_classifier.fit(X_train, y_train)
    accuracy_level = voting_classifier.score(X_test, y_test)
    print('accuracy_level:', accuracy_level,'\n')
    predict = voting_classifier.predict(X_test)
    print('prediction:', Counter(predict), '\n')
    return accuracy_level


machineLearnig_step('ZBRA')


