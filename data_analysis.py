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

data = pd.read_csv("merged-table.csv")
data = data.drop('Unnamed: 0', axis = 1)
data.date = pd.to_datetime(data.date)

holiday = pd.read_csv("data/holidays.csv")

dates = pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='7D')

holiday_list = holiday['date'].tolist()

is_holiday_list = pd.DataFrame(
    data={"is_holiday": [int(date in holiday_list) for date in data['date']]})

data['is_holiday'] = is_holiday_list

data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
data['quarter'] = data['date'].dt.quarter

ohe = OneHotEncoder()
season_encoded = ohe.fit_transform(data[["season_type"]])
season_encoded = season_encoded.toarray()
data = pd.concat([data, pd.DataFrame(season_encoded, columns=['Autumn-Winter', 'Summer-Spring'])], axis=1)


ohe = OneHotEncoder()
promotion_encoded = ohe.fit_transform(data[["promotion_type"]])
promotion_encoded = promotion_encoded.toarray()
promotions = ["BlackFriday", "General", "Main-1", "Main-2", "Main-3", "No promo", "SeasonMiddle-1", "SeasonMiddle-2"]
data = pd.concat([data, pd.DataFrame(promotion_encoded, columns = promotions)], axis=1)



X = data.drop('sales_amount', axis=1)
y = data['sales_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                random_state=0, train_size= 0.8)


