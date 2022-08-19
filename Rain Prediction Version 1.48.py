## Importing Libraries ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xlrd
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.linear_model import TweedieRegressor


## Creating dataframe ##
df = pd.read_excel('Sample_GaugeData.xlsx', sheet_name='RN',
     engine='openpyxl').iloc[4:-1,0:20].melt(id_vars=["GMT"])


## Removing all Na values ##
df = df[df['value'].notna()]

## Converting values to float ##
df.value = df.value.astype(float)


## Processing data - Might be some small errors in the calculations ##
df['DayMax'] = df.groupby(df['GMT'].dt.date)['value'].transform('max')
df['DayAvg'] = df.groupby(df['GMT'].dt.date)['value'].transform(np.mean)*288
df['DayMin'] = df.groupby(df['GMT'].dt.date)['value'].transform('min')
df['weekMax'] = df.groupby(df['GMT'].dt.isocalendar().week)['value'].transform('max')
df['weektotal'] = df.groupby(df['GMT'].dt.isocalendar().week)['value'].transform(np.mean)*2016


## Creating function for encoding data ##
def encode_data(feature_name):
    dicts = {}
    unique_values = list(df[feature_name].unique())
    for i in range(len(unique_values)):
        dicts[unique_values[i]] = i
    return dicts

## Encoding data ##
df['GMT'].replace(encode_data('GMT'),inplace = True)
df["variable"].replace(encode_data("variable"),inplace = True)

## Defining Y and X values ##
X = df.drop(["DayAvg",'GMT','variable'], axis=1) #all columns except PRCP and RAIN, as both give the answer away.
y = df["DayAvg"]

## Creating train and test models ##
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


## Creating a regression class ##
LR = TweedieRegressor(alpha=0.0, link='auto', power=1)

## Fitting the training data ##
LR.fit(X_train,y_train)

## Creating prediction
y_prediction = LR.predict(X_test)

## Removing all negative values ##
y_prediction[y_prediction < 0.0] = 0.0

## Creating dataframe containing actualy vs predicted results ##
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_prediction})


## Predicting the accuracy score ##
score=r2_score(y_test,y_prediction)
print("R Score is ",score)
print("mean_sqrd_error is==",mean_squared_error(y_test,y_prediction))
print("root_mean_squared error of is==",np.sqrt(mean_squared_error(y_test,y_prediction)))

## Creating a graph containing first 60 initial values ##
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(df1.index[0:60],df1.Actual[0:60].values,color='g')
ax1.scatter(df1.index[0:60],df1.Predicted[0:60].values, color='r')