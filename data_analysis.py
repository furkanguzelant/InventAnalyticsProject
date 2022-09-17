# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 14:49:21 2022

@author: Casper
"""

import pandas as pd
import math
from statsmodels.tsa.stattools import adfuller
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
import datetime
from sklearn.metrics import mean_squared_log_error

data = pd.read_csv("data.csv")
test = pd.read_csv('test_final.csv')

holiday = pd.read_csv("data/holidays.csv")

df = pd.concat([data, test], axis = 0)

df = df.reset_index()


df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1)

# test['promotion_type'] = test['promotion_type'].fillna('No promotion')
# dates = pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='7D')

holiday_list = holiday['date'].tolist()

is_holiday_list = pd.DataFrame(
    data={"is_holiday": [int(date in holiday_list) for date in df['date']]})

df['is_holiday'] = is_holiday_list

df.date = pd.to_datetime(df.date)
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['quarter'] = df['date'].dt.quarter

ohe = OneHotEncoder()
season_encoded = ohe.fit_transform(df[["season_type"]])
season_encoded = season_encoded.toarray()
df = pd.concat([df, pd.DataFrame(season_encoded, columns=['Autumn-Winter', 'Summer-Spring'])], axis=1)


ohe = OneHotEncoder()
promotion_encoded = ohe.fit_transform(df[["promotion_type"]])
promotion_encoded = promotion_encoded.toarray()
promotions = ["No promo", "BlackFriday", "General", "Main-1", "Main-2", "Main-3", "SeasonMiddle-1", "SeasonMiddle-2"]
df = pd.concat([df, pd.DataFrame(promotion_encoded, columns = promotions)], axis=1)


df = df.sort_values(by='date')
df= df.drop(['index', 'date', 'season_type', 'promotion_type'], axis=1)

train_df = df.iloc[:405292]
test_df = df.iloc[405292:]




X = train_df.drop('sales_amount', axis=1)
test_df = test_df.drop('sales_amount', axis=1)
y = train_df['sales_amount']


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                random_state=0, shuffle = False, train_size= 0.8)

model = lgb.LGBMRegressor(first_metric_only = True)

model.fit(X_train, y_train)

forecast = model.predict(X_test)

forecast = abs(forecast)

print('  LightGBM rmse: %.4f' % mean_squared_log_error(y_test, forecast))

model.fit(X, y)
forecast_test = model.predict(test_df)

result = pd.DataFrame(forecast_test, columns=['sales_amount'])
result['id'] = result.index + 1
result = result[result.columns[::-1]]


result.to_csv('submission.csv', index=False)


